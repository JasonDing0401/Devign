project="combined"
python -u main.py --dataset ${project} --input_dir ${project} \
--feature_size 225 --model_type ggnn \
2>&1 | tee logs/combined_$(date "+%m.%d-%H.%M.%S").log