import argparse
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def fetch_malayalam_data(
    target_gb: float = 1.0,
    output_path: str = str(Path("data") / "input.txt"),
    dataset_name: str = "ai4bharat/IndicCorpV2",
    dataset_config: str = "indiccorp_v2",
    lang_split: str = "mal_Mlym",
    data_dir: Optional[str] = None,
    text_field: str = "text",
    max_records: Optional[int] = None,
) -> None:
    target_bytes = int(target_gb * 1024 * 1024 * 1024)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data_dir:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            data_dir=data_dir,
            split="train",
            streaming=True,
        )
    else:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=lang_split,
            streaming=True,
        )

    current_bytes = 0
    num_records = 0

    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Downloading")

    with out_path.open("w", encoding="utf-8") as f:
        for entry in ds:
            if max_records is not None and num_records >= max_records:
                break

            text = entry.get(text_field, "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            line = text + "\n"
            b = line.encode("utf-8")
            f.write(line)

            current_bytes += len(b)
            num_records += 1
            pbar.update(len(b))

            if current_bytes >= target_bytes:
                break

    pbar.close()

    print(f"wrote: {os.path.abspath(str(out_path))}")
    print(f"bytes: {current_bytes}")
    print(f"records: {num_records}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-gb", type=float, default=1.0)
    p.add_argument("--output", type=str, default=str(Path("data") / "input.txt"))
    p.add_argument("--dataset", type=str, default="ai4bharat/IndicCorpV2")
    p.add_argument("--config", type=str, default="indiccorp_v2")
    p.add_argument("--lang", type=str, default="mal_Mlym")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    fetch_malayalam_data(
        target_gb=args.target_gb,
        output_path=args.output,
        dataset_name=args.dataset,
        dataset_config=args.config,
        lang_split=args.lang,
        data_dir=args.data_dir,
        text_field=args.text_field,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
