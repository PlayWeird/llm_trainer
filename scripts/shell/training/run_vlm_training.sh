#!/bin/bash

# VLM Training Script
# This script processes the Flickr8k dataset and trains a Vision-Language Model
# on the processed data using DeepSpeed for distributed training.

# Get absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"

# Default parameters
INPUT_DIR="${PROJECT_ROOT}/datasets/test_dataset/vlm/flickr8k"
PROCESSED_DIR="${PROJECT_ROOT}/datasets/processed/vlm/flickr8k"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/gemma-3-27b-vlm-finetuned"
MODEL_NAME="google/gemma-3-27b-it"
SAMPLE_SIZE=100  # Small sample size for testing, increase for real training
NUM_EPOCHS=3
BATCH_SIZE=1
ACCUM_STEPS=16
LEARNING_RATE=2e-5
USE_LORA=true
USE_4BIT=true

# DeepSpeed configuration
DS_CONFIG="${PROJECT_ROOT}/configs/training/ds_config_zero3.json"

# Flags for controlling script flow
PROCESS_DATASET=true
RUN_TRAINING=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --accum-steps)
      ACCUM_STEPS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --ds-config)
      DS_CONFIG="$2"
      shift 2
      ;;
    --skip-processing)
      PROCESS_DATASET=false
      shift
      ;;
    --skip-training)
      RUN_TRAINING=false
      shift
      ;;
    --no-lora)
      USE_LORA=false
      shift
      ;;
    --no-4bit)
      USE_4BIT=false
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input-dir DIR         Input directory containing raw Flickr8k data (default: ${INPUT_DIR})"
      echo "  --processed-dir DIR     Directory to save processed data (default: ${PROCESSED_DIR})"
      echo "  --output-dir DIR        Output directory for training results (default: ${OUTPUT_DIR})"
      echo "  --model-name NAME       Model name or path (default: ${MODEL_NAME})"
      echo "  --sample-size SIZE      Number of examples to include (default: ${SAMPLE_SIZE})"
      echo "  --num-epochs N          Number of training epochs (default: ${NUM_EPOCHS})"
      echo "  --batch-size SIZE       Batch size per GPU (default: ${BATCH_SIZE})"
      echo "  --accum-steps STEPS     Gradient accumulation steps (default: ${ACCUM_STEPS})"
      echo "  --learning-rate RATE    Learning rate (default: ${LEARNING_RATE})"
      echo "  --ds-config FILE        DeepSpeed config file (default: ${DS_CONFIG})"
      echo "  --skip-processing       Skip dataset processing"
      echo "  --skip-training         Skip model training"
      echo "  --no-lora               Disable LoRA fine-tuning"
      echo "  --no-4bit               Disable 4-bit quantization"
      echo "  --help                  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure conda environment is active
if [ -z "${CONDA_DEFAULT_ENV}" ] || [ "${CONDA_DEFAULT_ENV}" != "llm_trainer_env" ]; then
  echo "Activating llm_trainer_env conda environment..."
  
  # Check if conda is available as a command
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llm_trainer_env
  else
    # Try to find and source conda.sh
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "${HOME}/anaconda3")
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate llm_trainer_env
  fi
  
  # Verify activation
  if [ $? -ne 0 ]; then
    echo "Error: Failed to activate llm_trainer_env conda environment"
    echo "Please create and activate it manually with:"
    echo "conda env create -f ${PROJECT_ROOT}/environment.yml"
    echo "conda activate llm_trainer_env"
    exit 1
  fi
fi

# Check CUDA and GPU availability
echo "Checking GPU setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
  echo "Error: GPU check failed. Please ensure CUDA is properly configured."
  exit 1
fi

# Create output directories
mkdir -p "${PROCESSED_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Process dataset if needed
if [ "${PROCESS_DATASET}" = true ]; then
  echo "Processing Flickr8k dataset..."
  python "${PROJECT_ROOT}/scripts/preprocessing/process_flickr8k.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${PROCESSED_DIR}" \
    --sample_size "${SAMPLE_SIZE}" \
    --val_split 0.1 \
    --test_split 0.1 \
    --seed 42
  
  if [ $? -ne 0 ]; then
    echo "Error: Dataset processing failed."
    exit 1
  fi
  
  echo "Dataset processing completed successfully."
fi

# Run training if needed
if [ "${RUN_TRAINING}" = true ]; then
  echo "Starting VLM training..."
  
  # Prepare training command
  TRAIN_CMD="python ${PROJECT_ROOT}/scripts/training/train_gemma3_vlm.py"
  
  # Add DeepSpeed if multiple GPUs are available
  GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
  if [ "${GPU_COUNT}" -gt 1 ]; then
    echo "Using DeepSpeed for multi-GPU training (${GPU_COUNT} GPUs)"
    TRAIN_CMD="deepspeed ${TRAIN_CMD} --deepspeed ${DS_CONFIG}"
  fi
  
  # Add common parameters
  TRAIN_CMD="${TRAIN_CMD} \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_path ${PROCESSED_DIR}/flickr8k_train.json \
    --image_dir ${PROCESSED_DIR}/images \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --logging_steps 10 \
    --save_steps 100"
  
  # Add LoRA if enabled
  if [ "${USE_LORA}" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} \
      --use_lora True \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05"
  else
    TRAIN_CMD="${TRAIN_CMD} --use_lora False"
  fi
  
  # Add 4-bit quantization if enabled
  if [ "${USE_4BIT}" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_4bit True"
  else
    TRAIN_CMD="${TRAIN_CMD} --use_4bit False"
  fi
  
  # Execute the training command
  echo "Executing: ${TRAIN_CMD}"
  eval "${TRAIN_CMD}"
  
  if [ $? -ne 0 ]; then
    echo "Error: Training failed."
    exit 1
  fi
  
  echo "Training completed successfully."
  echo "Model saved to: ${OUTPUT_DIR}"
fi

echo "All processes completed."