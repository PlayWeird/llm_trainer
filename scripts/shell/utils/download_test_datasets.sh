#!/bin/bash

# Script to download test datasets for LLM and VLM training
# This script activates the conda environment and runs the dataset download script

set -e  # Exit on any error

# Activate conda environment if not already activated
if [[ -z "${CONDA_DEFAULT_ENV}" || "${CONDA_DEFAULT_ENV}" != "gemma3_env" ]]; then
    echo "Activating gemma3_env conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gemma3_env
fi

# Default parameters
LLM_DATASET="dolly"
VLM_DATASET="flickr8k"
LLM_SAMPLES=2000
VLM_SAMPLES=100  # Reduced sample size for faster testing
DOWNLOAD_TYPE="both"  # Can be "both", "llm", or "vlm"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --llm)
            LLM_DATASET="$2"
            shift
            shift
            ;;
        --vlm)
            VLM_DATASET="$2"
            shift
            shift
            ;;
        --llm-samples)
            LLM_SAMPLES="$2"
            shift
            shift
            ;;
        --vlm-samples)
            VLM_SAMPLES="$2"
            shift
            shift
            ;;
        --llm-only)
            DOWNLOAD_TYPE="llm"
            shift
            ;;
        --vlm-only)
            DOWNLOAD_TYPE="vlm"
            shift
            ;;
        --help)
            echo "Usage: ./shell/utils/download_test_datasets.sh [options]"
            echo ""
            echo "Options:"
            echo "  --llm <dataset>        Specify LLM dataset (dolly, oasst) [default: dolly]"
            echo "  --vlm <dataset>        Specify VLM dataset (flickr8k, vqa-rad) [default: flickr8k]"
            echo "  --llm-samples <num>    Number of LLM examples to download [default: 2000]"
            echo "  --vlm-samples <num>    Number of VLM examples to download [default: 500]"
            echo "  --llm-only             Download only LLM dataset"
            echo "  --vlm-only             Download only VLM dataset"
            echo "  --help                 Show this help message"
            echo ""
            echo "Example:"
            echo "  ./shell/utils/download_test_datasets.sh --llm dolly --vlm flickr8k --llm-samples 15000 --vlm-samples 8000"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Prepare the command
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
CMD="python $PROJECT_ROOT/scripts/preprocessing/download_test_datasets.py"
CMD="$CMD --llm_dataset $LLM_DATASET --vlm_dataset $VLM_DATASET"
CMD="$CMD --llm_samples $LLM_SAMPLES --vlm_samples $VLM_SAMPLES"

# Add the appropriate flags for download type
if [ "$DOWNLOAD_TYPE" == "llm" ]; then
    CMD="$CMD --llm_only"
elif [ "$DOWNLOAD_TYPE" == "vlm" ]; then
    CMD="$CMD --vlm_only"
fi

# Print the command to be executed
echo "Executing: $CMD"
echo "This may take a while depending on dataset size and your internet connection..."
echo ""

# Execute the command
eval $CMD

echo ""
echo "✅ Dataset download complete!"
echo "You can now use these datasets for model testing and training."