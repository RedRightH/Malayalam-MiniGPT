import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from model import MalayalamGPT, MalayalamGPTConfig
from tokenizer import load_tokenizer


def _maybe_summary(model: torch.nn.Module, input_ids: torch.Tensor) -> None:
    try:
        from torchinfo import summary  # type: ignore
    except Exception:
        print("torchinfo not installed; skipping layer-wise summary")
        return

    try:
        s = summary(model, input_data=(input_ids,), depth=6, verbose=1)
        print(s)
    except Exception as e:
        print(f"torchinfo summary failed: {e}")


def _maybe_torchview_graph(model: torch.nn.Module, input_ids: torch.Tensor, out_path: Path) -> None:
    try:
        from torchview import draw_graph  # type: ignore
    except Exception:
        print("torchview not installed; skipping architecture graph export")
        return

    try:
        g = draw_graph(model, input_data=(input_ids,), expand_nested=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # torchview can output via graphviz under the hood
        g.visual_graph.render(str(out_path.with_suffix("")), format=out_path.suffix.lstrip("."), cleanup=True)
        print(f"saved architecture graph: {out_path}")
    except Exception as e:
        print(f"torchview graph export failed: {e}")


def _plot_attention(att: torch.Tensor, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed; cannot plot attention")
        return

    # att: (heads, T, T)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    im = ax.imshow(att, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_attention_heatmaps(
    model: MalayalamGPT,
    input_ids: torch.Tensor,
    out_dir: Path,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
) -> None:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = input_ids.device
    input_ids = input_ids.to(model_device)
    model.eval()
    with torch.no_grad():
        _logits, _loss, attn_layers = model(input_ids, return_attn=True)

    # attn_layers: List[Tensor] each (B, heads, T, T)
    t = int(input_ids.shape[1])
    layer_ids = list(range(len(attn_layers))) if layers is None else layers

    for li in layer_ids:
        if li < 0 or li >= len(attn_layers):
            continue

        att = attn_layers[li][0].detach().float().cpu()  # (heads, T, T)
        head_ids = list(range(att.shape[0])) if heads is None else heads

        # mean over heads
        mean_att = att.mean(dim=0).numpy()  # (T, T)
        _plot_attention(mean_att, out_dir / f"layer_{li:02d}_mean.png", title=f"Layer {li} attention (mean over heads), T={t}")

        for hi in head_ids:
            if hi < 0 or hi >= att.shape[0]:
                continue
            h_att = att[hi].numpy()
            _plot_attention(h_att, out_dir / f"layer_{li:02d}_head_{hi:02d}.png", title=f"Layer {li} head {hi} attention, T={t}")


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--tokenizer", type=str, default=str(Path("tokenizer_models") / "malayalam-bpe.json"))
    p.add_argument("--ckpt", type=str, default=str(Path("out") / "ckpt_best.pt"))
    p.add_argument("--prompt", type=str, default="കേരളം")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-tokens", type=int, default=64)

    p.add_argument("--arch-summary", action="store_true")
    p.add_argument("--arch-graph", type=str, default=None)

    p.add_argument("--attn-out", type=str, default=str(Path("out") / "viz" / "attn"))
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--heads", type=int, nargs="*", default=None)

    args = p.parse_args()

    device = torch.device(args.device)

    tok = load_tokenizer(args.tokenizer)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("config", None)
    if not isinstance(cfg_dict, dict):
        raise SystemExit("checkpoint missing config")

    cfg = MalayalamGPTConfig(**cfg_dict)
    model = MalayalamGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    enc = tok.encode(args.prompt, add_special_tokens=False)
    ids = enc.ids[: args.max_tokens]
    if len(ids) == 0:
        raise SystemExit("prompt produced empty tokenization")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    if args.arch_summary:
        _maybe_summary(model, input_ids)

    if args.arch_graph:
        _maybe_torchview_graph(model, input_ids, Path(args.arch_graph))

        # Some graphing backends may move/clone tensors internally; re-pin to requested device.
        model = model.to(device)
        input_ids = input_ids.to(device)

    out_dir = Path(args.attn_out)
    save_attention_heatmaps(
        model=model,
        input_ids=input_ids,
        out_dir=out_dir,
        layers=args.layers,
        heads=args.heads,
    )
    print(f"saved attention heatmaps under: {out_dir}")


if __name__ == "__main__":
    main()
