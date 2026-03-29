import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    if special_tokens is None:
        special_tokens = list(DEFAULT_SPECIAL_TOKENS)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.train(files, trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is not None and eos_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> $B:1 <eos>:1",
            special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
        )

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_path)


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_lines(
    tokenizer: Tokenizer,
    lines: Iterable[str],
    add_special_tokens: bool,
) -> List[List[int]]:
    enc = tokenizer.encode_batch(list(lines), add_special_tokens=add_special_tokens)
    return [e.ids for e in enc]


def decode_ids(tokenizer: Tokenizer, ids: List[int], skip_special_tokens: bool) -> str:
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--files", nargs="+", default=None)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--tokenizer-path", type=str, default=str(Path("tokenizer_models") / "malayalam-bpe.json"))
    p.add_argument("--encode", type=str, default=None)
    p.add_argument("--decode", type=str, default=None)
    p.add_argument("--add-special-tokens", action="store_true")
    p.add_argument("--skip-special-tokens", action="store_true")
    args = p.parse_args()

    if args.train:
        if not args.files:
            raise SystemExit("--files is required when --train is set")
        tok = train_bpe_tokenizer(
            files=args.files,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        save_tokenizer(tok, args.tokenizer_path)
        print(f"saved tokenizer to {args.tokenizer_path}")
        print(f"vocab_size={tok.get_vocab_size()}")
        return

    tok_path = Path(args.tokenizer_path)
    if not tok_path.exists():
        raise SystemExit(
            f"tokenizer not found at {tok_path}. Run: python tokenizer.py --train --files data/input.txt"
        )

    tok = load_tokenizer(str(tok_path))

    if args.encode is not None:
        e = tok.encode(args.encode, add_special_tokens=args.add_special_tokens)
        print(" ".join(str(i) for i in e.ids))
        return

    if args.decode is not None:
        ids = [int(x) for x in args.decode.strip().split() if x]
        text = decode_ids(tok, ids, skip_special_tokens=args.skip_special_tokens)
        print(text)
        return

    print(f"loaded tokenizer from {tok_path}")
    print(f"vocab_size={tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
