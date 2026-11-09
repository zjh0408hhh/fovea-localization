# 3. 五折训练和断点恢复
IMPLEMENTATION = 'context_flip_from_scratch_v1'
# 签名不含种子列表，增加种子时可以复用已训练的模型
training_config = {key: value for key, value in CFG.items() if key != 'seeds'}
signature_data = dict(implementation=IMPLEMENTATION, cfg=training_config, ids=gt['id'].tolist(),
                      coordinates=coords.tolist(),
                      images=[hashlib.sha256(i['image'].tobytes()).hexdigest() for i in train_items],
                      boxes=[i['polygon'].tolist() if i['polygon'] is not None else None for i in train_items],
                      test_ids=[i['id'] for i in test_items],
                      test_images=[hashlib.sha256(i['image'].tobytes()).hexdigest() for i in test_items])
signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()[:16]


def cpu_state(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def average_states(states):
    if not states:
        raise ValueError('没有固定轮次快照')
    result = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        result[key] = torch.stack(tensors).float().mean(0).to(tensors[0].dtype) if tensors[0].is_floating_point() else tensors[-1].clone()
    return result


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def capture_rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                cuda=torch.cuda.get_rng_state_all() if AMP else [])


def restore_rng(state):
    random.setstate(state['python'])
    x = state['numpy']
    np.random.set_state((x[0], np.asarray(x[1], np.uint32), x[2], x[3], x[4]))
    torch.set_rng_state(state['torch'])
    if AMP:
        for device_id, value in enumerate(state['cuda'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value, device_id)




def run_fold(seed, fold, val_idx):
    tr_idx = np.array(sorted(set(range(len(gt))) - set(val_idx.tolist())), int)
    if HAS_NEGATIVE and not (visible[tr_idx].any() and (~visible[tr_idx]).any()):
        raise ValueError('训练折缺少可见/不可见类别，请减少折数或采用分层分组划分。')
    stem = 'context_s%d_f%d_%s' % (seed, fold, signature)
    final_path, last_path = OUT / (stem + '.pt'), OUT / (stem + '_last.pt')
    history_path = OUT / (stem + '_history.csv')
    best_path = OUT / (stem + '_best.pt')
    if final_path.is_file():
        done = torch.load(str(final_path), map_location='cpu', weights_only=True)
        if (done['signature'] != signature or done['val_indices'] != val_idx.tolist()
                or done['seed'] != seed or done['fold'] != fold):
            raise ValueError('已完成权重与当前任务不匹配')
        print('Loaded seed=%d fold=%d/%d final_RMSE=%.4f' % (seed, fold + 1, CFG['folds'], done['val_rmse']), flush=True)
        return np.asarray(done['val_predictions'], np.float32), np.asarray(done['test_predictions'], np.float32), dict(
            seed=seed, fold=fold + 1, rmse=done['val_rmse'], best_epoch=done['best_epoch'],
            best_epoch_rmse=done['best_epoch_rmse'], path=str(final_path))
    seed_everything(seed + 97 * fold)
    model = ContextFoveaNet().to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    loader = make_loader(train_items, tr_idx, training=True)
    total_steps = CFG['epochs'] * len(loader)
    warmup = min(4 * len(loader), max(1, total_steps // 10))

    def lr_factor(step):
        if step < warmup:
            return max(.05, (step + 1) / warmup)
        fraction = min(1., (step - warmup) / max(1, total_steps - warmup))
        floor = CFG['min_lr'] / CFG['lr']
        return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history, snapshots, start_epoch = [], [], 1
    best_score, best_epoch = float('inf'), 0
    if last_path.is_file():
        state = torch.load(str(last_path), map_location='cpu', weights_only=True)
        if state['signature'] != signature or state['val_indices'] != val_idx.tolist():
            raise ValueError('断点与当前配置/验证折不匹配')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, snapshots = state['history'], state['snapshots']
        best_score, best_epoch = state['best_score'], state['best_epoch']
        start_epoch = int(state['epoch']) + 1
        restore_rng(state['rng'])
        del state
        print('Resume seed=%d fold=%d/%d epoch=%d' % (seed, fold + 1, CFG['folds'], start_epoch), flush=True)
    for epoch in range(start_epoch, CFG['epochs'] + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(DEVICE, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP):
                loss = paired_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失出现非有限值')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
                update_ema(ema, model)
            losses.append(float(loss.detach().cpu()))
        # 每轮记录验证分数，仍按计划训练到最后一轮
        val_pred, _ = predict(ema, train_items, val_idx)
        score = rmse(val_pred, coords[val_idx])
        if not math.isfinite(score):
            raise FloatingPointError('验证RMSE出现非有限值')
        if score < best_score:
            best_score, best_epoch = score, epoch
            # 单独备份验证分数最低的权重，提交仍用固定轮次的平均权重
            atomic_save(dict(signature=signature, implementation=IMPLEMENTATION, config=training_config,
                             seed=seed, fold=fold, val_indices=val_idx.tolist(), model=cpu_state(ema),
                             epoch=epoch, val_rmse=score, checkpoint_type='validation_best_candidate'), best_path)
        if epoch in CFG['tail_epochs']:
            snapshots.append(cpu_state(ema))
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_rmse=score,
                   best_rmse=best_score, best_epoch=best_epoch,
                   lr=float(optimizer.param_groups[0]['lr']))
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print('Seed=%d Fold=%d/%d epoch=%03d/%03d loss=%.4f val_RMSE=%.4f best_RMSE=%.4f lr=%.7f' %
              (seed, fold + 1, CFG['folds'], epoch, CFG['epochs'], row['loss'], score, best_score, row['lr']), flush=True)
        if epoch % CFG['checkpoint_every'] == 0 or epoch == CFG['epochs']:
            atomic_save(dict(signature=signature, val_indices=val_idx.tolist(), epoch=epoch,
                             model=cpu_state(model), ema=cpu_state(ema), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                             history=history, snapshots=snapshots, best_score=best_score,
                             best_epoch=best_epoch, rng=capture_rng()), last_path)
    if len(snapshots) != len(CFG['tail_epochs']):
        raise RuntimeError('固定轮次快照不完整，请检查断点')
    final_state = average_states(snapshots)
    model.load_state_dict(final_state)
    val_pred, _ = predict(model, train_items, val_idx)
    test_pred, _ = predict(model, test_items)
    final_score = rmse(val_pred, coords[val_idx])
    atomic_save(dict(signature=signature, implementation=IMPLEMENTATION, config=training_config,
                     seed=seed, fold=fold, val_indices=val_idx.tolist(), model=final_state,
                     averaged_epochs=CFG['tail_epochs'], val_rmse=final_score,
                     best_epoch=best_epoch, best_epoch_rmse=best_score,
                     val_predictions=val_pred.tolist(), test_predictions=test_pred.tolist()), final_path)
    # 本折训练完后删除优化器断点
    if last_path.is_file():
        last_path.unlink()
    record = dict(seed=seed, fold=fold + 1, rmse=final_score, best_epoch=best_epoch,
                  best_epoch_rmse=best_score, path=str(final_path))
    print('Final seed=%d fold=%d/%d RMSE=%.4f' % (seed, fold + 1, CFG['folds'], final_score), flush=True)
    del model, ema, optimizer, scheduler, scaler, snapshots, loader, final_state
    gc.collect()
    if AMP:
        torch.cuda.empty_cache()
    return val_pred, test_pred, record


needed = len(CFG['seeds']) * CFG['folds']
available = sum((OUT / ('context_s%d_f%d_%s.pt' % (seed, fold, signature))).is_file()
                for seed in CFG['seeds'] for fold in range(CFG['folds']))
print('Models=%d Working_completed=%d To_train=%d' % (needed, available, needed - available), flush=True)
seed_oof, seed_test, model_records = [], [], []
for seed in CFG['seeds']:
    this_oof = np.full_like(coords, np.nan)
    this_test = []
    for fold, val_idx in enumerate(folds):
        val_pred, test_pred, record = run_fold(seed, fold, val_idx)
        this_oof[val_idx] = val_pred
        this_test.append(test_pred)
        model_records.append(record)
    assert np.isfinite(this_oof).all()
    seed_oof.append(this_oof)
    seed_test.append(np.mean(this_test, axis=0))
    print('Seed=%d OOF_RMSE=%.4f' % (seed, rmse(this_oof, coords)), flush=True)
oof = np.mean(seed_oof, axis=0)
test_prediction = np.mean(seed_test, axis=0)
