# Setup Guide

Reproduce the Qwen3.5-0.8B embedding model PoC from scratch on a fresh machine.

## 1. Python Environment

The project uses an existing virtualenv. On the original machine this was `/media/.venv`. On a new machine, create a fresh venv and install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Transformers must be installed from source (Qwen3.5 support not yet in a release)
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

pip install torch>=2.8.0 accelerate>=1.8.0 peft>=0.15.0 \
  mteb>=2.12.0 sentence-transformers>=3.0.0 qwen-vl-utils>=0.0.14 \
  datasets>=3.0.0 decord>=0.6.0 opencv-python-headless torchvision>=0.16.0 \
  deepspeed>=0.17.0 pillow numpy scipy

# Match torchaudio to your torch+CUDA version, e.g. for cu128:
pip install torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel; print('OK')"
python -c "import peft, mteb; print(peft.__version__, mteb.__version__)"
```

## 2. Model Downloads

### Qwen3.5-0.8B (base model for training)
```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```
Expected: ~1.7GB, produces `models/Qwen3.5-0.8B/` with `model.safetensors-00001-of-00001.safetensors`, `config.json`, `tokenizer.json`, etc.

### Qwen3-VL-Embedding-2B (reference baseline for eval comparison)
```bash
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir models/Qwen3-VL-Embedding-2B
```
Expected: ~4GB. Only needed to reproduce their MMTEB scores locally.

## 3. PoC Training Data

```bash
source .venv/bin/activate
python datasets/download_poc_data.py --max_per_source 10000
```
Downloads AllNLI, MS-MARCO, GooAQ, SimpleWiki triplets (40K samples total) into `datasets/poc_data/poc_train.jsonl`.

For a larger run:
```bash
python datasets/download_poc_data.py --max_per_source 100000
```

## 4. Run PoC Training

Single GPU (RTX 4090 / A100):
```bash
CUDA_VISIBLE_DEVICES=0 python src/train/train_embedding.py \
  --model_path models/Qwen3.5-0.8B \
  --data_path datasets/poc_data/poc_train.jsonl \
  --output_dir outputs/poc-v1 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 1 \
  --gradient_checkpointing \
  --use_mrl \
  --log_interval 5 \
  --save_steps 100
```

Key parameters:
- LoRA rank 64 / alpha 128 on all linear projections (43M trainable / 896M total = 4.83%)
- InfoNCE loss with in-batch negatives, temperature 0.02
- MRL (Matryoshka Representation Learning) at dims [64, 128, 256, 512, 1024]
- bf16 mixed precision, gradient checkpointing enabled
- ~7 min per epoch on 2000 samples with RTX 4090

## 5. Evaluation

### Quick eval (single task, ~10 seconds):
```bash
# Untrained base model
python src/eval/run_mmteb.py --model_path models/Qwen3.5-0.8B \
  --output_dir results/qwen35-base --tasks BIOSSES

# LoRA checkpoint (auto-detects adapter_config.json)
python src/eval/run_mmteb.py --model_path outputs/poc-v1/final \
  --output_dir results/poc-v1 --tasks BIOSSES

# Qwen3-VL-Embedding-2B baseline (requires model download)
python src/eval/run_mmteb.py --model_path models/Qwen3-VL-Embedding-2B \
  --output_dir results/qwen3vl-2b --tasks BIOSSES
```

### Fast MMTEB subset (9 tasks, one per type, ~30 min):
```bash
python src/eval/run_mmteb.py --model_path outputs/poc-v1/final \
  --output_dir results/poc-v1-fast
```

### Full MMTEB (all English tasks, several hours):
```bash
python src/eval/run_mmteb.py --model_path outputs/poc-v1/final \
  --output_dir results/poc-v1-full --full
```

### Compare two models:
```bash
python src/eval/compare_models.py --results_a results/qwen3vl-2b --results_b results/poc-v1
```

## 6. PoC Results (from original machine)

| Model | BIOSSES STS |
|-------|------------|
| Qwen3.5-0.8B (untrained) | 31.73 |
| Qwen3.5-0.8B (PoC, 2K samples) | **76.84** |
| Qwen3-VL-Embedding-2B (reported) | 74.29 |

Training log for 2K sample run:
- Loss: 1.56 -> 0.25 over 62 steps
- Time: ~7 minutes on single RTX 4090
- LoRA: 43M trainable params (4.83% of 896M)

## 7. Architecture Notes

Qwen3.5-0.8B uses a hybrid DeltaNet architecture (not standard transformer):
- 24 layers: 3:1 ratio of Gated DeltaNet (linear attention) to standard GQA
- Hidden dim: 1024 (output embedding dimension)
- DeltaNet layers cannot be made bidirectional; we use causal attention + last-token pooling (same as the Qwen3-VL-Embedding paper)
- LoRA targets 12 projection types: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj`
- Embedding template: `<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{content}<|im_end|><|endoftext|>` -- last token pooled as embedding

## 8. Dataset Catalog

`datasets/training_datasets.csv` contains 38 datasets (text + multimodal) with HuggingFace URLs, sizes, and training stage assignments. Use this for scaling up training on cloud.
