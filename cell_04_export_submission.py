# 4. 按样例顺序生成提交文件
def build_submission(template, items, prediction):
    prediction = np.asarray(prediction)
    if prediction.shape != (len(items), 2) or not np.isfinite(prediction).all():
        raise ValueError('测试预测形状或数值有误')
    mapping = {item['id']: xy for item, xy in zip(items, prediction)}
    values = []
    for ident in template['ImageID']:
        match = re.fullmatch(r'(.+)_Fovea_([XY])', str(ident).strip())
        if match is None:
            raise ValueError('提交ID格式错误: ' + str(ident))
        key, axis = numeric_id(match.group(1)), 0 if match.group(2) == 'X' else 1
        values.append(float(mapping[key][axis]))
    result = template.copy()
    result['value'] = values
    assert len(result) == 2 * len(items)
    assert result['ImageID'].tolist() == template['ImageID'].tolist()
    assert result['ImageID'].is_unique and np.isfinite(result['value']).all()
    return result


submission = build_submission(sample, test_items, test_prediction)
submission_path = OUT / ('submission_competition_only_SMOKE_ONLY.csv' if SMOKE_TEST else 'submission_competition_only.csv')
submission.to_csv(submission_path, index=False)
oof_table = gt[['id', 'Fovea_X', 'Fovea_Y']].copy()
oof_table['fold'] = 0
for fold, idx in enumerate(folds):
    oof_table.loc[idx, 'fold'] = fold + 1
oof_table['pred_X'], oof_table['pred_Y'] = oof[:, 0], oof[:, 1]
oof_table.to_csv(OUT / 'oof_competition_only.csv', index=False)
pd.DataFrame(model_records).to_csv(OUT / 'competition_only_model_scores.csv', index=False)
seed_records = [dict(seed=seed, oof_rmse=rmse(pred, coords)) for seed, pred in zip(CFG['seeds'], seed_oof)]
pd.DataFrame(seed_records).to_csv(OUT / 'competition_only_seed_scores.csv', index=False)
final_folds = [dict(fold=f + 1, rmse=rmse(oof[idx], coords[idx])) for f, idx in enumerate(folds)]
pd.DataFrame(final_folds).to_csv(OUT / 'competition_only_fold_scores.csv', index=False)
per_seed_test = []
for seed, prediction in zip(CFG['seeds'], seed_test):
    for item, xy in zip(test_items, prediction):
        per_seed_test.append(dict(seed=seed, id=item['id'], Fovea_X=float(xy[0]), Fovea_Y=float(xy[1])))
pd.DataFrame(per_seed_test).to_csv(OUT / 'competition_only_seed_test.csv', index=False)
summary = dict(implementation=IMPLEMENTATION, signature=signature, config=CFG,
               oof_rmse=rmse(oof, coords), seeds=seed_records, folds=final_folds,
               models=model_records, checkpoint_selection='fixed_epoch_EMA_weight_average',
               has_real_invisible_examples=HAS_NEGATIVE, smoke_test=SMOKE_TEST,
               external_data=False, external_pretrained_weights=False,
               reads_weights_from_input=False, needs_prior_submission=False,
               submission_path=str(submission_path))
(OUT / 'competition_only_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('Ensemble OOF_RMSE=%.4f' % summary['oof_rmse'], flush=True)
print('提交文件: ' + str(submission_path), flush=True)
