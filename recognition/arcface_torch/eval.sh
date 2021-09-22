CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
--model-prefix work_dirs/ms1mv3_r18/backbone.pth \
--image-path IJB_release/IJBC \
--result-dir work_dirs/ms1mv3_r18 \
--batch-size 128 \
--job ms1mv3_arcface_r18 \
--target IJBC \
--network r18