#!/usr/bin/env bash
# Unified contrastive pretraining launch script (Stage 1 / s0).
#
# Trains Qwen3.5-0.8B from scratch on ~9.2M mixed-modality examples using:
#   - Full 5-term InfoNCE (q-q, d-d, q-d, hard neg, positive) + false-neg masking
#   - Cross-GPU embedding gathering (contrastive_batch = 1024 total)
#   - GradCache (auto-enabled: per_device=512 > micro=4)
#   - MRL on 3 dims: [1024, 256, 64]
#   - Image budget: 1,280 tokens (~1.31M pixels), matching paper Section 5.2
#   - LoRA rank=32, alpha=32
#
# For 2 GPUs:  per_device = 512, 128 micro-batches of 4
# For 8 GPUs:  per_device = 128,  32 micro-batches of 4
# Just change CUDA_DEVICES to scale.

export CUDA_DEVICES=0,1
export HF_TOKEN=$(cat /home/jupyter/shared/embeddings/.hf_token_local | tr -d '\n')
export USE_WANDB=true
export WANDB_PROJECT=embeddings
export WANDB_ENTITY=radi-and-people
export WANDB_RUN_NAME=qwen35-0.8b-10M-pretrain-cbs1024
export IMAGE_DIR=/home/jupyter/shared/embeddings/datasets/mmeb_train_images/images
export DATA_DIR=/data/training_data
export OUTPUT_DIR=/data/outputs/qwen35-0.8b-10M-pretrain
export MODEL_PATH=/home/jupyter/shared/embeddings/models/Qwen3.5-0.8B

# --- Batch size controls ---
export CONTRASTIVE_BATCH=1024
export MICRO_BATCH=4
export GRAD_ACCUM=1

export EPOCHS=1
export LR=2e-5
export MAX_LENGTH=512
export MAX_PIXELS=1310720
export MRL_DIMS=1024,256,64
export TRAINING_STAGE=1
export SAVE_STEPS=500
export LOG_INTERVAL=1
export NUM_WORKERS=4
export GRADIENT_CHECKPOINTING=true

cd /home/jupyter/shared/embeddings
exec bash scripts/run_training.sh
