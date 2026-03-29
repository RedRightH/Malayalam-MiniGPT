import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_rows(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows", [])
    out_rows: List[Dict[str, float]] = []
    for r in rows:
        out_rows.append(
            {
                "temperature": float(r["temperature"]),
                "top_k": float(r["top_k"]),
                "distinct_1": float(r["distinct_1"]),
                "distinct_2": float(r["distinct_2"]),
                "entropy_nats": float(r["entropy_nats"]),
                "entropy_norm": float(r["entropy_norm"]),
            }
        )
    return obj, out_rows


def _best_by(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
    best = rows[0]
    for r in rows[1:]:
        if r[key] > best[key]:
            best = r
    return best


def _format_topk(v: float) -> str:
    if v < 0:
        return "None"
    return str(int(v))


def _make_table(rows: List[Dict[str, float]], best_keys: List[str]) -> str:
    best_for: Dict[str, Tuple[float, float]] = {}
    for k in best_keys:
        b = _best_by(rows, k)
        best_for[k] = (b["temperature"], b["top_k"])

    header = "| temp | top_k | distinct_1 | distinct_2 | entropy_norm | entropy_nats |\n|---:|---:|---:|---:|---:|---:|"
    lines = [header]

    for r in sorted(rows, key=lambda x: (x["temperature"], x["top_k"])):
        temp = r["temperature"]
        topk = r["top_k"]
        d1 = f"{r['distinct_1']:.6f}"
        d2 = f"{r['distinct_2']:.6f}"
        en = f"{r['entropy_norm']:.6f}"
        ea = f"{r['entropy_nats']:.6f}"

        if best_for.get("distinct_1") == (temp, topk):
            d1 = f"**{d1}**"
        if best_for.get("distinct_2") == (temp, topk):
            d2 = f"**{d2}**"
        if best_for.get("entropy_norm") == (temp, topk):
            en = f"**{en}**"
        if best_for.get("entropy_nats") == (temp, topk):
            ea = f"**{ea}**"

        lines.append(
            f"| {temp:.3f} | {_format_topk(topk)} | {d1} | {d2} | {en} | {ea} |"
        )

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(Path("out") / "sweeps" / "sweep.json"))
    p.add_argument("--out-md", type=str, default=None)
    args = p.parse_args()

    inp = Path(args.input)
    obj, rows = _load_rows(inp)

    table = _make_table(rows, best_keys=["distinct_1", "distinct_2", "entropy_norm", "entropy_nats"])

    ckpt = obj.get("checkpoint", "")
    prompts = obj.get("prompts", [])
    sample_tokens = obj.get("sample_tokens", None)

    report_lines = [
        "# Sweep Summary",
        f"checkpoint: `{ckpt}`" if ckpt else "",
        f"sample_tokens: `{sample_tokens}`" if sample_tokens is not None else "",
        f"prompts: `{prompts}`" if prompts else "",
        "",
        table,
        "",
        "## Best settings (per metric)",
    ]

    for key in ("distinct_1", "distinct_2", "entropy_norm", "entropy_nats"):
        b = _best_by(rows, key)
        report_lines.append(
            f"- **{key}**: temp={b['temperature']:.3f}, top_k={_format_topk(b['top_k'])}, value={b[key]:.6f}"
        )

    report = "\n".join([ln for ln in report_lines if ln != ""]).strip() + "\n"
    print(report)

    if args.out_md:
        outp = Path(args.out_md)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
