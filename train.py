import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from model import MalayalamGPT, MalayalamGPTConfig
from tokenizer import load_tokenizer


def _read_meta(meta_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not meta_path.exists():
        return meta
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def _dtype_from_meta(meta: Dict[str, str]) -> np.dtype:
    name = meta.get("dtype", "uint16")
    if name == "uint16":
        return np.uint16
    if name == "uint32":
        return np.uint32
    return np.uint16


class BinDataset:
    def __init__(self, bin_path: Path, dtype: np.dtype) -> None:
        if not bin_path.exists():
            raise FileNotFoundError(str(bin_path))
        self.bin_path = bin_path
        self.dtype = dtype
        self.data = np.memmap(str(bin_path), dtype=dtype, mode="r")

    def get_batch(self, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(self.data.shape[0])
        if n <= block_size + 1:
            raise ValueError("Dataset too small for the requested block_size")

        ix = np.random.randint(0, n - block_size - 1, size=(batch_size,))
        x = np.stack([self.data[i : i + block_size] for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + block_size] for i in ix])

        x_t = torch.from_numpy(x.astype(np.int64, copy=False)).to(device)
        y_t = torch.from_numpy(y.astype(np.int64, copy=False)).to(device)
        return x_t, y_t


def cosine_lr(it: int, warmup_iters: int, lr: float, min_lr: float, max_iters: int) -> float:
    if it < warmup_iters:
        return lr * (it + 1) / max(1, warmup_iters)
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: MalayalamGPT,
    train_data: BinDataset,
    val_data: BinDataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}

    for split_name, ds in ("train", train_data), ("val", val_data):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = ds.get_batch(batch_size=batch_size, block_size=block_size, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model did not return loss")
            losses[k] = loss
        out[split_name] = float(losses.mean().item())

    model.train()
    return out


@torch.no_grad()
def sample_text(
    model: MalayalamGPT,
    tokenizer_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> str:
    tok = load_tokenizer(str(tokenizer_path))
    enc = tok.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([enc.ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    ids = out[0].tolist()
    return tok.decode(ids, skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tokenizer", type=str, default=str(Path("tokenizer_models") / "malayalam-bpe.json"))
    p.add_argument("--out-dir", type=str, default="out")

    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--sample-prompt", type=str, default="കേരളം")
    p.add_argument("--sample-tokens", type=int, default=200)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--sample-top-k", type=int, default=50)

    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    tok = load_tokenizer(str(Path(args.tokenizer)))
    tok_vocab = int(tok.get_vocab_size())
    if args.vocab_size != tok_vocab:
        print(f"overriding --vocab-size={args.vocab_size} with tokenizer vocab_size={tok_vocab}")
        args.vocab_size = tok_vocab

    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    meta = _read_meta(data_dir / "meta.txt")
    np_dtype = _dtype_from_meta(meta)

    train_data = BinDataset(train_bin, dtype=np_dtype)
    val_data = BinDataset(val_bin, dtype=np_dtype)

    cfg = MalayalamGPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    model = MalayalamGPT(cfg).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_dtype: Optional[torch.dtype] = None
    if args.amp:
        if device.type == "cuda":
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.bfloat16

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=tuple(args.betas),
        device_type=device.type,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    iter_num = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = int(ckpt.get("iter_num", 0))
        best_val = float(ckpt.get("best_val", best_val))

    t0 = time.time()

    while iter_num < args.max_iters:
        lr = cosine_lr(
            it=iter_num,
            warmup_iters=args.warmup_iters,
            lr=args.learning_rate,
            min_lr=args.min_lr,
            max_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
                amp_dtype=amp_dtype,
            )

            val_loss = losses["val"]
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = out_dir / "ckpt_best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val": best_val,
                        "config": asdict(cfg),
                    },
                    ckpt_path,
                )

            ckpt_path = out_dir / "ckpt_last.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val": best_val,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )

            text = sample_text(
                model=model,
                tokenizer_path=Path(args.tokenizer),
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_tokens,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
                device=device,
            )

            print(
                f"iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}\n{text}\n"
            )

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(args.grad_accum):
            x, y = train_data.get_batch(
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
            )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Model did not return loss")

            loss = loss / args.grad_accum
            loss_accum += float(loss.item())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"iter {iter_num}: loss {loss_accum:.4f}, lr {lr:.2e}, {dt:.3f}s/iter"
            )

        iter_num += 1


if __name__ == "__main__":
    main()
