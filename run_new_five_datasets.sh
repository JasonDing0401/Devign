project="new_five_datasets"

python -u main.py --dataset ${project} --input_dir \
/data3/dlvp_local_data/dataset_merged/new_five_datasets/ \
--feature_size 224 --model_type ggnn --batch_size 128 --train \
2>&1 | tee logs/${project}_$(date "+%m.%d-%H.%M.%S").log