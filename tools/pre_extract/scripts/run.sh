#!/bin/bash

# If on Slurm Cluster

# The following code divides the original data collection (say, /path/to/original.jsonl) into 32 splits,
# then uses 32 GPUs, one for each split, to pre-extract the vae latents
# The extracted pickle files are saved at /path/to/datasets/my_data/pre_extract/pickle
# The 32 new data collections (one from each GPU) are saved at /path/to/datasets/my_data/pre_extract/record
# You can use the `concat_record.py` script to concat the 32 sub-collections to one complete collection

srun -n32 --ntasks-per-node=8 --gres=gpu:8 \
python -u tools/pre_extract/pre_extract.py \
--splits=32 \
--in_filename /path/to/my_data.jsonl \
--pkl_dir /path/to/datasets/my_data/pre_extract/pickle \
--record_dir  /path/to/datasets/my_data/pre_extract/record \
--target_size 256 \
--target_fps 8


# Otherwise, if you are on a single node with 8 GPUs, please refer to the following
for i in {0..7}
do
  export CUDA_VISIBLE_DEVICES=${i}
  python -u tools/pre_extract/pre_extract.py \
  --splits=8 \
  --rank_bias=${i} \
  --in_filename /path/to/my_data.jsonl \
  --pkl_dir /path/to/datasets/my_data/pre_extract/pickle \
  --record_dir  /path/to/datasets/my_data/pre_extract/record \
  --target_size 256 \
  --target_fps 8
  sleep 2s
done
