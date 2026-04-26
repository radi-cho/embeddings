#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into its base model and save.

Usage:
  python scripts/merge_lora.py <ckpt_dir> [out_dir]

If out_dir is omitted, saves to <parent>/merged-<step>.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.qwen35_embedding import Qwen35ForEmbedding  # noqa: E402


def main():
    ckpt = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if out is None:
        step = ckpt.name.replace("checkpoint-", "")
        out = ckpt.parent / f"merged-{step}"

    cfg_path = ckpt / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No adapter_config.json in {ckpt}")
    base_path = json.loads(cfg_path.read_text())["base_model_name_or_path"]
    print(f"Loading base from {base_path}", flush=True)
    base = Qwen35ForEmbedding.from_pretrained(
        base_path, torch_dtype=torch.bfloat16)
    print(f"Loading adapter from {ckpt}", flush=True)
    peft = PeftModel.from_pretrained(base, str(ckpt))
    print("Merging", flush=True)
    merged = peft.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out}", flush=True)
    merged.save_pretrained(out, safe_serialization=True)

    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "processor_config.json", "special_tokens_map.json"]:
        src = ckpt / fn
        if src.exists():
            shutil.copy2(src, out / fn)
        else:
            src2 = Path(base_path) / fn
            if src2.exists():
                shutil.copy2(src2, out / fn)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
