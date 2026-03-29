# Malayalam-MiniGPT

A compact, **GPT-2 / nanoGPT-style causal Transformer** implemented in plain PyTorch and trained on Malayalam text.

This repo includes:
- A from-scratch GPT-like model (`model.py`) with causal self-attention, MLP blocks, residual connections, and weight tying.
- Data fetching + preprocessing to build a `.bin` token dataset.
- Training (`train.py`) and evaluation/sampling + sweep utilities (`eval.py`, `summarize_sweep.py`).
- Visualization utilities for **attention heatmaps** and an **architecture graph** (`visualize_model.py`).

## Project structure
- `model.py`
  - `MalayalamGPT` (Transformer language model)
  - `MalayalamGPTConfig`
- `tokenizer.py`
  - Train/load a ByteLevel BPE tokenizer (HuggingFace `tokenizers`)
- `fetch_data.py`
  - Stream and save Malayalam text from HuggingFace datasets
- `data_prep.py`
  - Tokenize and write `train.bin` / `val.bin` + `meta.txt`
- `train.py`
  - Train from scratch (or resume from a checkpoint)
- `eval.py`
  - Evaluate checkpoint (loss/ppl/acc) + generate samples + optional sweeps
- `visualize_model.py`
  - Export attention maps and (optionally) a model graph

## Setup

### Create environment
Use any Python env (Conda/venv). Minimum: Python 3.9+ recommended.

### Install dependencies
This project doesn’t ship with a `requirements.txt` yet, so install the basics manually:

```bash
pip install torch numpy tokenizers datasets tqdm
```

Optional (evaluation extras):
```bash
pip install evaluate
```

Optional (visualization):
```bash
pip install matplotlib torchinfo torchview graphviz
```

Graph rendering note (Windows): `torchview` requires the **Graphviz system executable** `dot`.
- If you use conda:
  ```bash
  conda install -c conda-forge graphviz
  ```
- Verify:
  ```bash
  dot -V
  ```

## Data (not included in Git)
Large artifacts are intentionally excluded via `.gitignore`.
- `data/` (e.g., `input.txt`, `train.bin`, `val.bin`) is ignored.
- `out/` checkpoints and training outputs are ignored.

You need to generate your own data locally.

## Workflow

### 1) Fetch text data
Example:
```bash
python fetch_data.py --target-gb 1.0 --output data/input.txt
```

### 2) Train or load tokenizer
A tokenizer JSON is included under `tokenizer_models/`.
If you want to re-train it:
```bash
python tokenizer.py --train --files data/input.txt --tokenizer-path tokenizer_models/malayalam-bpe.json
```

### 3) Prepare binary dataset
```bash
python data_prep.py --input data/input.txt --tokenizer tokenizer_models/malayalam-bpe.json --out-dir data
```

This should produce:
- `data/train.bin`
- `data/val.bin`
- `data/meta.txt`

### 4) Train
```bash
python train.py --data-dir data --tokenizer tokenizer_models/malayalam-bpe.json --out-dir out
```

Resume training from a saved checkpoint:
```bash
python train.py --resume out/ckpt_last.pt
```

### 5) Evaluate / sample
```bash
python eval.py --ckpt out/ckpt_best.pt --data-dir data --tokenizer tokenizer_models/malayalam-bpe.json
```

## Visualize the model

### Attention heatmaps (layer/head)
This exports PNG heatmaps under `out/viz/attn/`.

Example (layer 0, head 0):
```bash
python visualize_model.py \
  --ckpt out/ckpt_best.pt \
  --tokenizer tokenizer_models/malayalam-bpe.json \
  --prompt "കേരളം" \
  --device cpu \
  --max-tokens 32 \
  --layers 0 \
  --heads 0
```

### Architecture graph (SVG)
```bash
python visualize_model.py --arch-graph out/viz/arch_graph.svg --device cpu --ckpt out/ckpt_best.pt --tokenizer tokenizer_models/malayalam-bpe.json
```

### Layer-wise summary
```bash
python visualize_model.py --arch-summary --device cpu --ckpt out/ckpt_best.pt --tokenizer tokenizer_models/malayalam-bpe.json
```

## Notes
- This is a **from-scratch Transformer implementation** in the Karpathy/nanoGPT style (PyTorch modules, custom attention block), not a wrapper around pretrained Hugging Face models.
- Checkpoints are local by default (ignored by git). If you want to publish weights, consider Git LFS or GitHub Releases.
