# MMMOS: Multi-domain Multi-axis Audio Quality Assessment

- Presented at **IEEE ASRU 2025**

## Dataset
The training and validation dataset used in this work is located in the `audiomos2025_track2` directory.

## Experiments
To run the training scripts, you should:

1. Update all paths to your own environment and load **WavLM** and **M2d** checkpoints.  
2. Configure the loss function using the `args` parameter:  
   - `1`: UTMOS loss  
   - `2`: Con loss  
   - `3`: DCQ loss  
   - `4`: CCC loss  
3. Run `train.py` directly with the appropriate arguments.
