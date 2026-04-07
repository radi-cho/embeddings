#!/usr/bin/env bash
export CUDA_DEVICES=0,1
export HF_TOKEN=$(cat /home/jupyter/shared/embeddings/.hf_token_local | tr -d '\n')
export USE_WANDB=true
export WANDB_PROJECT=embeddings
export WANDB_ENTITY=radi-and-people
export WANDB_RUN_NAME=qwen35-0.8b-mmeb-simple-b32
export IMAGE_DIR=/home/jupyter/shared/embeddings/datasets/mmeb_train_images/images
export OUTPUT_DIR=/data/outputs/qwen35-0.8b-mmeb-stage1
export MODEL_PATH=/home/jupyter/shared/embeddings/models/Qwen3.5-0.8B
# batch=32 should fit with gradient checkpointing (~15GB vs 40GB available)
export BATCH_SIZE=32
export EFFECTIVE_BATCH=512
export EPOCHS=1
export LR=1e-4
export MAX_LENGTH=512
export MAX_PIXELS=401408
export MRL_DIMS=1024,256,64
export SAVE_STEPS=500
export LOG_INTERVAL=1
# more workers to keep GPU fed
export NUM_WORKERS=8
export NO_GRAD_CACHE=true
export GRAD_ACCUM=1
export GRADIENT_CHECKPOINTING=true
cd /home/jupyter/shared/embeddings
exec bash scripts/run_training.sh
