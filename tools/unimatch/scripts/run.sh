#!/bin/bash

# Before running this script, you should have finished the vae latent pre-extraction process,
# and used the `concat_record.py` script to obtain the complete data collection file.
# Let's assume the new collection file is now at
# /path/to/datasets/my_data/pre_extract/record/record.jsonl

# If on Slurm Cluster

# The following code divides the data collection file from the pre-extraction step
# (/path/to/datasets/my_data/pre_extract/record/record.jsonl) into 32 splits,
# then uses 32 GPUs, one for each split, to compute the motion score using the UniMatch model.
# The 32 new data collections (one from each GPU) with motion score will be saved at
# /path/to/datasets/my_data/UniMatch/record/
# You can use the `concat_record.py` script to concat the 32 sub-collections to one complete collection

srun -n32 --ntasks-per-node=8 --gres=gpu:8 \
python -u tools/unimatch/compute.py \
--splits=32 \
--in_filename /path/to/datasets/my_data/pre_extract/record/record.jsonl \
--record_dir  /path/to/datasets/my_data/UniMatch/record/ \


# Otherwise, if you are on a single node with 8 GPUs, please refer to the following

for i in {0..7}
do
  export CUDA_VISIBLE_DEVICES=${i}
  python -u tools/unimatch/compute.py \
  --splits=8 \
  --rank_bias=${i} \
  --in_filename /path/to/datasets/my_data/pre_extract/record/record.jsonl \
  --record_dir  /path/to/datasets/my_data/UniMatch/record/ \
  sleep 2s
done
