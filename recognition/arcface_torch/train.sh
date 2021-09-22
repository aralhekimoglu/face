# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1235 train.py configs/ms1mv3_r18

## global_step is increased after every batch of data is loaded.
## So for 200 data points, after one epoch it should be 200/16=12

## need to update num ids in config using rec2img.py code.

python3 train.py configs/ms1mv3_r18