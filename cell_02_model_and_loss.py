# 2. 图像增强、定位模型和损失函数
def augment_item(image, xy, polygon, vis):
    center = (S - 1) / 2
    for _ in range(12):
        matrix = cv2.getRotationMatrix2D((center, center), random.uniform(-12, 12),
                                         random.uniform(.9, 1.1)).astype(np.float32)
        matrix[:, 2] += np.random.uniform(-.03, .03, 2).astype(np.float32) * S
        if random.random() < .5:
            matrix[0] *= -1
            matrix[0, 2] += S - 1
        point = transform_points([xy], matrix)[0]
        if not vis or ((point > .03 * S).all() and (point < .97 * S).all()):
            break
    else:
        matrix = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        point = xy.copy()
    image = cv2.warpAffine(image, matrix, (S, S), flags=cv2.INTER_LINEAR)
    polygon = transform_points(polygon, matrix) if polygon is not None else None
    image = image.astype(np.float32) / 255
    image = np.power(image, random.uniform(.85, 1.18))
    image *= np.random.uniform(.92, 1.08, (1, 1, 3)).astype(np.float32)
    image = (image - .5) * random.uniform(.9, 1.1) + .5
    if random.random() < .15:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(.3, .7))
    if random.random() < .2:
        image += np.random.normal(0, .005, image.shape).astype(np.float32)
    return np.clip(image * 255, 0, 255).astype(np.uint8), point, polygon, matrix


def image_tensor(image):
    green = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(image[:, :, 1])
    x = np.concatenate([image, green[:, :, None]], axis=2).astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def coordinate_metric(base_inverse, augmentation):
    # 将增强后的坐标误差换回原图像素，再除以pixel_unit
    return (base_inverse[:, :2] @ np.linalg.inv(augmentation[:, :2]) *
            ((S - 1) / CFG['pixel_unit'])).astype(np.float32)


class FoveaDataset(Dataset):
    def __init__(self, items, indices=None, training=False):
        self.items = items
        self.indices = list(range(len(items))) if indices is None else list(indices)
        self.training = training

    def __len__(self):
        return len(self.indices) * (CFG['repeats'] if self.training else 1)

    def __getitem__(self, index):
        idx = self.indices[index % len(self.indices)]
        item = self.items[idx]
        if not self.training:
            return dict(image=image_tensor(item['image']), index=idx)
        image, xy, polygon = item['image'], item['xy'].copy(), item['polygon']
        image, xy, polygon, matrix = augment_item(image, xy, polygon, item['visible'])
        region = np.zeros((HS, HS), np.float32)
        if polygon is not None:
            pts = np.round(polygon * (HS - 1) / (S - 1)).astype(np.int32)
            cv2.fillConvexPoly(region, pts, 1)
        return dict(image=image_tensor(image), xy=torch.tensor(xy / (S - 1), dtype=torch.float32),
                    region=torch.from_numpy(region[None]), visible=torch.tensor(item['visible']),
                    metric=torch.from_numpy(coordinate_metric(item['inverse'], matrix)))


def make_loader(items, indices=None, training=False):
    return DataLoader(FoveaDataset(items, indices, training), batch_size=CFG['batch'],
                      shuffle=training, num_workers=CFG['workers'], pin_memory=AMP, drop_last=False)


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, co), nn.SiLU(inplace=True),
                                    nn.Conv2d(co, co, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, co), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    def __init__(self, ci, skip, co):
        super().__init__()
        self.conv = Block(ci + skip, co)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ContextFoveaNet(nn.Module):
    def __init__(self):
        super().__init__()
        y, x = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing='ij')
        self.register_buffer('position', torch.stack([x, y])[None], persistent=False)
        self.stem = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1, bias=False),
                                  nn.GroupNorm(8, 16), nn.SiLU(inplace=True))
        self.e1, self.e2, self.e3, self.e4 = Block(16, 16), Block(16, 32), Block(32, 64), Block(64, 128)
        self.mid = Block(128, 192)
        self.u4, self.u3 = Up(192, 128, 128), Up(128, 64, 64)
        self.u2, self.u1 = Up(64, 32, 32), Up(32, 16, 16)
        self.local_heat = nn.Conv2d(16, 1, 1)
        self.region = nn.Conv2d(16, 1, 1)
        self.context = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                     nn.Linear(192 * 4 * 4, 256), nn.SiLU(), nn.Dropout(.1),
                                     nn.Linear(256, CS * CS))
        self.presence = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        nn.init.normal_(self.local_heat.weight, std=.01)
        nn.init.zeros_(self.local_heat.bias)

    def forward(self, image):
        # 翻转后也用当前图像的坐标网格
        position = self.position.expand(len(image), -1, -1, -1).to(dtype=image.dtype)
        a = self.e1(self.stem(torch.cat([image, position], dim=1)))
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        d = self.e4(F.avg_pool2d(c, 2))
        m = self.mid(F.avg_pool2d(d, 2))
        z = self.u1(self.u2(self.u3(self.u4(m, d), c), b), a)
        coarse = self.context(m).reshape(-1, 1, CS, CS)
        heat = self.local_heat(z) + F.interpolate(coarse, size=z.shape[-2:], mode='bilinear', align_corners=True)
        return heat, coarse, self.region(z), self.presence(m).squeeze(1)


def make_grid(side):
    y, x = torch.meshgrid(torch.linspace(0, 1, side), torch.linspace(0, 1, side), indexing='ij')
    return torch.stack([x, y], dim=-1).reshape(1, side * side, 2).to(DEVICE)


GRID, COARSE_GRID = make_grid(HS), make_grid(CS)


def spatial_probability(heat):
    return heat.float().flatten(1).softmax(dim=1)


def decode(prob):
    return (prob.unsqueeze(-1) * GRID).sum(dim=1)


def heat_ce(heat, xy, vis, grid, sigma, side):
    target = (-(grid - xy[:, None]).square().sum(-1) / (2 * (sigma / (side - 1)) ** 2)).softmax(1)
    if vis.any():
        return -(target[vis] * heat.float().flatten(1).log_softmax(1)[vis]).sum(1).mean()
    return heat.float().sum() * 0


def supervised_loss(output, batch):
    heat, coarse, region, presence = output
    xy, vis = batch['xy'].float(), batch['visible'] > .5
    loss = heat_ce(heat, xy, vis, GRID, CFG['heat_sigma'], HS)
    loss = loss + .3 * heat_ce(coarse, xy, vis, COARSE_GRID, CFG['coarse_sigma'], CS)
    if vis.any():
        error = torch.einsum('bij,bj->bi', batch['metric'].float(), decode(spatial_probability(heat)) - xy)[vis]
        loss = loss + .5 * F.smooth_l1_loss(error, torch.zeros_like(error)) + .1 * error.square().mean()
    region, mask = region.float(), batch['region'].float()
    loss = loss + .1 * F.binary_cross_entropy_with_logits(region, mask, weight=1 + 7 * mask)
    p = region.sigmoid()
    dice = 1 - ((2 * (p * mask).sum((1, 2, 3)) + 1) /
                (p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean()
    loss = loss + .1 * dice
    if HAS_NEGATIVE:
        loss = loss + .25 * F.binary_cross_entropy_with_logits(presence.float(), batch['visible'].float())
    return loss


def paired_loss(model, batch):
    output = model(batch['image'])
    loss = supervised_loss(output, batch)
    count = min(CFG['pair_count'], len(batch['image']))
    if count:
        flip = model(batch['image'][:count].flip(-1))
        aligned = (flip[0].flip(-1), flip[1].flip(-1), flip[2].flip(-1), flip[3])
        subset = {key: value[:count] for key, value in batch.items()}
        loss = loss + CFG['pair_supervision'] * supervised_loss(aligned, subset)
        p, q = spatial_probability(output[0][:count]), spatial_probability(aligned[0])
        middle = ((p + q) / 2).clamp_min(1e-12)
        js = .5 * (p * (p.clamp_min(1e-12).log() - middle.log())).sum(1).mean()
        js = js + .5 * (q * (q.clamp_min(1e-12).log() - middle.log())).sum(1).mean()
        loss = loss + CFG['js_weight'] * js
        delta = torch.einsum('bij,bj->bi', subset['metric'].float(), decode(p) - decode(q))
        valid = subset['visible'] > .5
        if valid.any():
            loss = loss + CFG['consistency_weight'] * F.smooth_l1_loss(delta[valid], torch.zeros_like(delta[valid]))
    return loss


@torch.no_grad()
def update_ema(ema, model):
    for dst, src in zip(ema.parameters(), model.parameters()):
        dst.mul_(CFG['ema']).add_(src, alpha=1 - CFG['ema'])


@torch.no_grad()
def predict(model, items, indices=None):
    order = list(range(len(items))) if indices is None else list(indices)
    model.eval()
    all_xy, all_presence = [], []
    for batch in make_loader(items, order):
        x = batch['image'].to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP):
            heat, _, _, presence = model(x)
            heat_f, _, _, presence_f = model(x.flip(-1))
            prob = (spatial_probability(heat) + spatial_probability(heat_f.flip(-1))) / 2
            pv = (presence.float().sigmoid() + presence_f.float().sigmoid()) / 2 if HAS_NEGATIVE else torch.ones(len(x), device=DEVICE)
        all_xy.append(decode(prob).cpu().numpy())
        all_presence.append(pv.cpu().numpy())
    normalized, pv = np.concatenate(all_xy), np.concatenate(all_presence)
    result = []
    for point, idx in zip(normalized, order):
        item = items[idx]
        point = transform_points([point * (S - 1)], item['inverse'])[0]
        result.append(np.clip(point, [0, 0], item['wh'] - 1))
    xy = np.asarray(result, np.float32)
    if HAS_NEGATIVE:
        xy[pv < .5] = 0
    return xy, pv


def rmse(pred, target):
    return float(np.sqrt(np.mean((np.asarray(pred, np.float64) - np.asarray(target, np.float64)) ** 2)))
