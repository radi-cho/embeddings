python src/train/train.py \
--model_path models/Qwen3-VL-2B \
--max_samples_per_subset 100 \
--batch_size 8 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--epochs 1 \
--lr 1e-4 \
--output_dir outputs/test-2B-mmeb-train-split \
--image_dir datasets/mmeb_cache/mmeb_v2_image_tasks/MMEB