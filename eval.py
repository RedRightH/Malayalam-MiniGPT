import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@torch.no_grad()
def eval_loss_ppl_acc(
    model: MalayalamGPT,
    ds: BinDataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    correct = 0
    total = 0

    for k in range(eval_iters):
        x, y = ds.get_batch(batch_size=batch_size, block_size=block_size, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Model did not return loss")
        losses[k] = loss

        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    loss_val = float(losses.mean().item())
    ppl = float(math.exp(min(50.0, loss_val)))
    acc = float(correct / max(1, total))
    return {"loss": loss_val, "perplexity": ppl, "accuracy": acc}


def _distinct_ngrams(tokens: List[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def _shannon_entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


@torch.no_grad()
def generate_from_prompts(
    model: MalayalamGPT,
    tokenizer_path: Path,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> List[str]:
    tok = load_tokenizer(str(tokenizer_path))
    out_texts: List[str] = []
    model.eval()

    for p in prompts:
        enc = tok.encode(p, add_special_tokens=False)
        idx = torch.tensor([enc.ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        ids = out[0].tolist()
        out_texts.append(tok.decode(ids, skip_special_tokens=True))

    return out_texts


@torch.no_grad()
def generate_ids_from_prompts(
    model: MalayalamGPT,
    tokenizer_path: Path,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> List[Tuple[List[int], int]]:
    tok = load_tokenizer(str(tokenizer_path))
    out_ids: List[Tuple[List[int], int]] = []
    model.eval()

    for p in prompts:
        enc = tok.encode(p, add_special_tokens=False)
        prompt_len = len(enc.ids)
        idx = torch.tensor([enc.ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        ids = out[0].tolist()
        out_ids.append((ids, prompt_len))

    return out_ids


def token_entropy_metrics(
    gens: List[Tuple[List[int], int]],
    vocab_size: int,
) -> Dict[str, float]:
    counts: Dict[int, int] = {}
    total_tokens = 0
    for ids, prompt_len in gens:
        cont = ids[prompt_len:]
        for t in cont:
            counts[t] = counts.get(t, 0) + 1
        total_tokens += len(cont)

    ent = _shannon_entropy_from_counts(counts)
    ent_per_token = ent
    norm = 0.0
    if vocab_size > 1:
        norm = ent / math.log(vocab_size)
    return {
        "gen_tokens": float(total_tokens),
        "entropy_nats": float(ent_per_token),
        "entropy_norm": float(norm),
    }


def _read_ref_file(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            continue
        p, r = line.split("\t", 1)
        p = p.strip()
        r = r.strip()
        if p and r:
            pairs.append((p, r))
    return pairs


def _maybe_bleu_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        import evaluate
    except Exception:
        return {}

    out: Dict[str, float] = {}

    try:
        bleu = evaluate.load("bleu")
        bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
        if isinstance(bleu_res, dict) and "bleu" in bleu_res:
            out["bleu"] = float(bleu_res["bleu"])
    except Exception:
        pass

    try:
        rouge = evaluate.load("rouge")
        rouge_res = rouge.compute(predictions=preds, references=refs)
        if isinstance(rouge_res, dict):
            for k in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
                if k in rouge_res:
                    out[k] = float(rouge_res[k])
    except Exception:
        pass

    return out


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, default=str(Path("out") / "ckpt_best.pt"))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tokenizer", type=str, default=str(Path("tokenizer_models") / "malayalam-bpe.json"))

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-iters", type=int, default=200)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--sample-prompts", type=str, nargs="*", default=["കേരളം", "ഇന്ന് ", "സർക്കാർ "])
    p.add_argument("--sample-tokens", type=int, default=200)
    p.add_argument("--sample-temperature", type=float, default=0.9)
    p.add_argument("--sample-top-k", type=int, default=50)

    p.add_argument("--sweep-temperatures", type=float, nargs="*", default=None)
    p.add_argument("--sweep-top-k", type=int, nargs="*", default=None)

    p.add_argument("--sweep-save-json", type=str, default=None)
    p.add_argument("--sweep-save-csv", type=str, default=None)
    p.add_argument("--sweep-plot", type=str, default=None)

    p.add_argument("--diversity-samples", type=int, default=20)

    p.add_argument("--ref-file", type=str, default=None)

    args = p.parse_args()

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but torch.cuda.is_available() is False; falling back to cpu")
        requested_device = "cpu"
    device = torch.device(requested_device)

    data_dir = Path(args.data_dir)
    meta = _read_meta(data_dir / "meta.txt")
    np_dtype = _dtype_from_meta(meta)

    val_ds = BinDataset(data_dir / "val.bin", dtype=np_dtype)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg_dict = ckpt.get("config", None)
    if not isinstance(cfg_dict, dict):
        raise SystemExit("checkpoint missing config")

    cfg = MalayalamGPTConfig(**cfg_dict)
    model = MalayalamGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    amp_dtype: Optional[torch.dtype] = None
    if args.amp:
        if device.type == "cuda":
            amp_dtype = torch.float16
        elif device.type == "cpu":
            amp_dtype = torch.bfloat16

    core = eval_loss_ppl_acc(
        model=model,
        ds=val_ds,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=cfg.block_size,
        device=device,
        amp_dtype=amp_dtype,
    )

    tok = load_tokenizer(str(Path(args.tokenizer)))
    vocab_size = int(tok.get_vocab_size())

    gen_prompts = list(args.sample_prompts)
    samples = generate_from_prompts(
        model=model,
        tokenizer_path=Path(args.tokenizer),
        prompts=gen_prompts,
        max_new_tokens=args.sample_tokens,
        temperature=args.sample_temperature,
        top_k=args.sample_top_k,
        device=device,
    )

    diversity_prompts = ["കേരളം" for _ in range(args.diversity_samples)]
    diversity_ids = generate_ids_from_prompts(
        model=model,
        tokenizer_path=Path(args.tokenizer),
        prompts=diversity_prompts,
        max_new_tokens=args.sample_tokens,
        temperature=1.0,
        top_k=args.sample_top_k,
        device=device,
    )

    diversity_texts: List[str] = []
    for ids, _prompt_len in diversity_ids:
        diversity_texts.append(tok.decode(ids, skip_special_tokens=True))

    all_ids: List[int] = []
    for ids, prompt_len in diversity_ids:
        all_ids.extend(ids[prompt_len:])

    distinct_1 = _distinct_ngrams(all_ids, 1)
    distinct_2 = _distinct_ngrams(all_ids, 2)

    ent_metrics = token_entropy_metrics(diversity_ids, vocab_size=vocab_size)

    advanced: Dict[str, float] = {}
    if args.ref_file:
        pairs = _read_ref_file(Path(args.ref_file))
        if pairs:
            prompts = [p for p, _ in pairs]
            refs = [r for _, r in pairs]
            preds = generate_from_prompts(
                model=model,
                tokenizer_path=Path(args.tokenizer),
                prompts=prompts,
                max_new_tokens=args.sample_tokens,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
                device=device,
            )
            advanced = _maybe_bleu_rouge(preds, refs)

    print("# Eval Report")
    print(f"checkpoint: {ckpt_path}")
    print(f"val_loss: {core['loss']:.6f}")
    print(f"perplexity: {core['perplexity']:.4f}")
    print(f"accuracy: {core['accuracy']:.6f}")
    print(f"distinct_1: {distinct_1:.6f}")
    print(f"distinct_2: {distinct_2:.6f}")
    print(f"prompt_entropy_nats: {ent_metrics['entropy_nats']:.6f}")
    print(f"prompt_entropy_norm: {ent_metrics['entropy_norm']:.6f}")

    if advanced:
        for k in sorted(advanced.keys()):
            print(f"{k}: {advanced[k]:.6f}")
    elif args.ref_file:
        print("bleu/rouge: skipped (install 'evaluate' to enable)")

    print("\n# Fixed Prompt Samples")
    for ptxt, stxt in zip(gen_prompts, samples):
        print(f"prompt: {ptxt}")
        print(stxt)
        print("---")

    if args.sweep_temperatures is not None or args.sweep_top_k is not None:
        temps = args.sweep_temperatures if args.sweep_temperatures is not None else [args.sample_temperature]
        topks: List[Optional[int]]
        if args.sweep_top_k is None:
            topks = [args.sample_top_k]
        else:
            topks = [None if k < 0 else int(k) for k in args.sweep_top_k]

        sweep_rows: List[Dict[str, float]] = []

        print("\n# Sampling Sweep")
        for t in temps:
            for k in topks:
                sweep_ids = generate_ids_from_prompts(
                    model=model,
                    tokenizer_path=Path(args.tokenizer),
                    prompts=gen_prompts,
                    max_new_tokens=args.sample_tokens,
                    temperature=float(t),
                    top_k=k,
                    device=device,
                )
                cont_ids: List[int] = []
                for ids, prompt_len in sweep_ids:
                    cont_ids.extend(ids[prompt_len:])

                d1 = _distinct_ngrams(cont_ids, 1)
                d2 = _distinct_ngrams(cont_ids, 2)
                ent = token_entropy_metrics(sweep_ids, vocab_size=vocab_size)

                sweep_rows.append(
                    {
                        "temperature": float(t),
                        "top_k": -1.0 if k is None else float(k),
                        "distinct_1": float(d1),
                        "distinct_2": float(d2),
                        "entropy_nats": float(ent["entropy_nats"]),
                        "entropy_norm": float(ent["entropy_norm"]),
                    }
                )

                print(f"temperature={float(t):.3f} top_k={k}")
                print(f"distinct_1={d1:.6f} distinct_2={d2:.6f} entropy_norm={ent['entropy_norm']:.6f}")
                for (ids, _pl), ptxt in zip(sweep_ids, gen_prompts):
                    print(f"prompt: {ptxt}")
                    print(tok.decode(ids, skip_special_tokens=True))
                    print("---")

        if args.sweep_save_json:
            outp = Path(args.sweep_save_json)
            outp.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint": str(ckpt_path),
                "vocab_size": vocab_size,
                "prompts": gen_prompts,
                "sample_tokens": args.sample_tokens,
                "rows": sweep_rows,
            }
            outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved sweep json: {outp}")

        if args.sweep_save_csv:
            outp = Path(args.sweep_save_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            header = ["temperature", "top_k", "distinct_1", "distinct_2", "entropy_nats", "entropy_norm"]
            lines = [",".join(header)]
            for r in sweep_rows:
                lines.append(
                    ",".join(
                        [
                            f"{r['temperature']}",
                            f"{r['top_k']}",
                            f"{r['distinct_1']}",
                            f"{r['distinct_2']}",
                            f"{r['entropy_nats']}",
                            f"{r['entropy_norm']}",
                        ]
                    )
                )
            outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"saved sweep csv: {outp}")

        if args.sweep_plot:
            try:
                import matplotlib.pyplot as plt
            except Exception:
                print("plot skipped (install matplotlib to enable)")
            else:
                outp = Path(args.sweep_plot)
                outp.parent.mkdir(parents=True, exist_ok=True)

                temps_v = [r["temperature"] for r in sweep_rows]
                topk_v = [r["top_k"] for r in sweep_rows]
                ent_v = [r["entropy_norm"] for r in sweep_rows]
                d1_v = [r["distinct_1"] for r in sweep_rows]
                d2_v = [r["distinct_2"] for r in sweep_rows]

                fig, ax = plt.subplots(1, 3, figsize=(16, 4), dpi=150)

                sc0 = ax[0].scatter(temps_v, topk_v, c=ent_v)
                ax[0].set_title("entropy_norm")
                ax[0].set_xlabel("temperature")
                ax[0].set_ylabel("top_k (-1 = None)")
                fig.colorbar(sc0, ax=ax[0])

                sc1 = ax[1].scatter(temps_v, topk_v, c=d1_v)
                ax[1].set_title("distinct_1")
                ax[1].set_xlabel("temperature")
                ax[1].set_ylabel("top_k (-1 = None)")
                fig.colorbar(sc1, ax=ax[1])

                sc2 = ax[2].scatter(temps_v, topk_v, c=d2_v)
                ax[2].set_title("distinct_2")
                ax[2].set_xlabel("temperature")
                ax[2].set_ylabel("top_k (-1 = None)")
                fig.colorbar(sc2, ax=ax[2])

                fig.tight_layout()
                fig.savefig(outp)
                plt.close(fig)
                print(f"saved sweep plot: {outp}")


if __name__ == "__main__":
    main()
