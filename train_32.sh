#!/bin/bash

# Training script for LoopedTransformer on graph connectivity prediction
# This script runs the training with all arguments explicitly specified

# Create output directory if it doesn't exist
NUM_NODES=32
OUTPUT_DIR="./outputs_${NUM_NODES}_nodes/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting training at $(date)" 
echo "Output directory: $OUTPUT_DIR" 

# Run the training script with all arguments specified
python train.py \
  --graph_size $NUM_NODES \
  --sample_p True \
  --p_range 0.02 0.1 \
  --add_self_loops True \
  --train_samples 50_000 \
  --val_samples 1_000 \
  --n_loop 5 \
  --hidden_size 256 \
  --tie_qk True \
  --read_in_method "zero_pad" \
  --num_attention_heads 1 \
  --layernorm_type "post" \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --min_learning_rate 1e-6 \
  --num_epochs 100 \
  --clip_grad_norm 1.0 \
  --use_auxiliary_loss False \
  --aux_loss_weight 0.0 \
  --seed 189763 \
  --output_dir $OUTPUT_DIR 

echo "Training completed at $(date)" 

# Copy this script to the output directory for reference
cp "$0" $OUTPUT_DIR/

# Print results summary
echo -e "\n=== Training Summary ===" 
echo "Results saved to: $OUTPUT_DIR" 
if [ -f "${OUTPUT_DIR}/best_model.pt" ]; then
  echo "Best model saved: ${OUTPUT_DIR}/best_model.pt"
fi
if [ -f "${OUTPUT_DIR}/training_curves.png" ]; then
  echo "Training curves: ${OUTPUT_DIR}/training_curves.png" 
fi
