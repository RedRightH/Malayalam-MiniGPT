import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np

from tokenizer import load_tokenizer, save_tokenizer, train_bpe_tokenizer


_MALAYALAM_BLOCK_RE = re.compile(r"[\u0D00-\u0D7F]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def filter_malayalam_text(text: str, keep_punct: bool) -> str:
    out_chars: List[str] = []
    for ch in text:
        if ch == "\n" or ch == "\t" or ch == " ":
            out_chars.append(ch)
            continue

        if _MALAYALAM_BLOCK_RE.match(ch):
            out_chars.append(ch)
            continue

        cat = unicodedata.category(ch)
        if keep_punct and (cat.startswith("P") or cat.startswith("S")):
            out_chars.append(ch)
            continue

    cleaned = "".join(out_chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(str(path))


def _select_token_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _iter_clean_lines(path: Path, keep_punct: bool):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = normalize_text(line)
            line = filter_malayalam_text(line, keep_punct=keep_punct).strip()
            if not line:
                continue
            yield line


def encode_to_temp_bin(
    tokenizer_path: Path,
    input_path: Path,
    tmp_bin_path: Path,
    keep_punct: bool,
    add_special_tokens: bool,
    batch_lines: int,
) -> Tuple[int, np.dtype]:
    tok = load_tokenizer(str(tokenizer_path))
    vocab_size = tok.get_vocab_size()
    dtype = _select_token_dtype(vocab_size)

    tmp_bin_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_bin_path.exists():
        tmp_bin_path.unlink()

    total_tokens = 0
    batch: List[str] = []

    for line in _iter_clean_lines(input_path, keep_punct=keep_punct):
        batch.append(line)
        if len(batch) >= batch_lines:
            enc = tok.encode_batch(batch, add_special_tokens=add_special_tokens)
            flat: List[int] = []
            for e in enc:
                flat.extend(e.ids)
            arr = np.asarray(flat, dtype=dtype)
            with tmp_bin_path.open("ab") as bf:
                arr.tofile(bf)
            total_tokens += int(arr.shape[0])
            batch = []

    if batch:
        enc = tok.encode_batch(batch, add_special_tokens=add_special_tokens)
        flat = []
        for e in enc:
            flat.extend(e.ids)
        arr = np.asarray(flat, dtype=dtype)
        with tmp_bin_path.open("ab") as bf:
            arr.tofile(bf)
        total_tokens += int(arr.shape[0])

    return total_tokens, dtype


def split_temp_to_train_val(
    tmp_bin_path: Path,
    train_path: Path,
    val_path: Path,
    dtype: np.dtype,
    val_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    n_tokens = tmp_bin_path.stat().st_size // np.dtype(dtype).itemsize
    if n_tokens < 10:
        raise ValueError("Not enough tokens to split")

    n_val = int(n_tokens * val_ratio)
    n_train = n_tokens - n_val

    train_path.parent.mkdir(parents=True, exist_ok=True)
    if train_path.exists():
        train_path.unlink()
    if val_path.exists():
        val_path.unlink()

    chunk_tokens = 1024 * 1024

    with tmp_bin_path.open("rb") as rf, train_path.open("ab") as tf, val_path.open("ab") as vf:
        remaining_train = n_train
        while remaining_train > 0:
            to_read = min(chunk_tokens, remaining_train)
            arr = np.fromfile(rf, dtype=dtype, count=to_read)
            if arr.size == 0:
                break
            arr.tofile(tf)
            remaining_train -= int(arr.size)

        while True:
            arr = np.fromfile(rf, dtype=dtype, count=chunk_tokens)
            if arr.size == 0:
                break
            arr.tofile(vf)

    return int(n_train), int(n_val)


def split_train_val(ids: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    n = ids.shape[0]
    if n < 10:
        raise ValueError("Not enough tokens to split")

    n_val = int(n * val_ratio)
    n_train = n - n_val
    return ids[:n_train], ids[n_train:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(Path("data") / "input.txt"))
    p.add_argument("--out-dir", type=str, default="data")
    p.add_argument("--tokenizer-path", type=str, default=str(Path("tokenizer_models") / "malayalam-bpe.json"))
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--keep-punct", action="store_true")
    p.add_argument("--add-special-tokens", action="store_true")
    p.add_argument("--batch-lines", type=int, default=2000)
    p.add_argument("--tmp-bin", type=str, default=None)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input text not found: {input_path}")

    tok_path = Path(args.tokenizer_path)
    if not tok_path.exists():
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok = train_bpe_tokenizer(
            files=[str(input_path)],
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        save_tokenizer(tok, str(tok_path))
        print(f"trained and saved tokenizer to {tok_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    tmp_path = Path(args.tmp_bin) if args.tmp_bin else (out_dir / "all.bin")

    total_tokens, dtype = encode_to_temp_bin(
        tokenizer_path=tok_path,
        input_path=input_path,
        tmp_bin_path=tmp_path,
        keep_punct=args.keep_punct,
        add_special_tokens=args.add_special_tokens,
        batch_lines=args.batch_lines,
    )

    train_tokens, val_tokens = split_temp_to_train_val(
        tmp_bin_path=tmp_path,
        train_path=train_path,
        val_path=val_path,
        dtype=dtype,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    meta_path = out_dir / "meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"input={os.path.abspath(str(input_path))}",
                f"tokenizer={os.path.abspath(str(tok_path))}",
                f"vocab_size={args.vocab_size}",
                f"min_frequency={args.min_frequency}",
                f"keep_punct={bool(args.keep_punct)}",
                f"add_special_tokens={bool(args.add_special_tokens)}",
                f"num_tokens={int(total_tokens)}",
                f"train_tokens={int(train_tokens)}",
                f"val_tokens={int(val_tokens)}",
                f"dtype={np.dtype(dtype).name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {train_path} ({train_tokens} tokens)")
    print(f"wrote {val_path} ({val_tokens} tokens)")


if __name__ == "__main__":
    main()
