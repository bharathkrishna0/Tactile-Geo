"""Train the small U-Net on the train split, select on validation, CPU only.

    python -m evaluation.unet.train [--steps 1500] [--base 16] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evaluation.unet.data import load, split_ids  # noqa: E402
from evaluation.unet.infer import predict  # noqa: E402
from evaluation.unet.model import SmallUNet  # noqa: E402

CROP = 256


def sample_batch(items, rng: np.random.Generator, batch: int):
    xs, ys = [], []
    for _ in range(batch):
        gray, mask, fg = items[rng.integers(len(items))]
        h, w = gray.shape
        if rng.random() < 0.6 and len(fg):
            cy, cx = fg[rng.integers(len(fg))]
            y = int(np.clip(cy - CROP // 2 + rng.integers(-64, 65), 0, h - CROP))
            x = int(np.clip(cx - CROP // 2 + rng.integers(-64, 65), 0, w - CROP))
        else:
            y, x = int(rng.integers(0, h - CROP + 1)), int(rng.integers(0, w - CROP + 1))
        g, m = gray[y:y + CROP, x:x + CROP], mask[y:y + CROP, x:x + CROP]
        if rng.random() < 0.5:
            g, m = g[:, ::-1], m[:, ::-1]
        g = np.clip(g * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08), 0, 1)
        xs.append(g.copy())
        ys.append(m.copy())
    return torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))[:, None]


def loss_fn(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(4.0)) + dice


def dice_on(model, items) -> float:
    model.eval()
    inter = total = 0.0
    for gray, mask, _ in items:
        pred = predict(model, gray) >= 0.5
        inter += float((pred & (mask > 0)).sum())
        total += float(pred.sum() + (mask > 0).sum())
    model.train()
    return 2 * inter / total if total else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "weights" / "unet_small.pt")
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def prep(ids):
        out = []
        for i in ids:
            _, gray, mask = load(i)
            out.append((gray, mask, np.argwhere(mask > 0)))
        return out

    train, val = prep(split_ids("train")), prep(split_ids("validation"))
    model = SmallUNet(base=args.base)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps)
    best, history, start = -1.0, [], time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train, rng, args.batch)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            d = dice_on(model, val)
            history.append({"step": step, "loss": round(float(loss), 4), "val_dice": round(d, 4),
                            "elapsed_s": round(time.perf_counter() - start, 1)})
            print(history[-1], flush=True)
            if d > best:
                best = d
                torch.save({"model": model.state_dict(), "base": args.base, "step": step, "val_dice": round(d, 4),
                            "train_images": len(train), "seed": args.seed, "crop": CROP,
                            "parameters": model.parameter_count}, args.out)
    (args.out.with_suffix(".history.json")).write_text(json.dumps(
        {"history": history, "best_val_dice": round(best, 4), "steps": args.steps, "batch": args.batch,
         "base": args.base, "parameters": model.parameter_count,
         "train_seconds": round(time.perf_counter() - start, 1)}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
