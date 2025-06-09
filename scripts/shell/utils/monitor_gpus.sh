#!/bin/bash
# Monitor GPU usage during training

echo "Monitoring GPU usage (press Ctrl+C to stop)..."
echo "Time | GPU 0 Mem | GPU 1 Mem | GPU 2 Mem | GPU 0 Util | GPU 1 Util | GPU 2 Util"
echo "-----+----------+-----------+-----------+------------+------------+-----------"

while true; do
    # Get current time
    TIME=$(date +"%H:%M:%S")
    
    # Get GPU memory usage
    GPU0_MEM=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU1_MEM=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU2_MEM=$(nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # Get GPU utilization
    GPU0_UTIL=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU1_UTIL=$(nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU2_UTIL=$(nvidia-smi --id=2 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    printf "%s | %5s MB | %5s MB | %5s MB | %10s%% | %10s%% | %10s%%\n" \
        "$TIME" "$GPU0_MEM" "$GPU1_MEM" "$GPU2_MEM" "$GPU0_UTIL" "$GPU1_UTIL" "$GPU2_UTIL"
    
    sleep 2
done