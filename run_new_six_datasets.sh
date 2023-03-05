project="new_six_datasets"

python -u main.py --dataset ${project} --input_dir \
/data3/dlvp_local_data/dataset_merged/${project}/ \
--feature_size 224 --model_type ggnn --batch_size 256 --train \
2>&1 | tee logs/${project}_$(date "+%m.%d-%H.%M.%S").log