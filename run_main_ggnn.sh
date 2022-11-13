project="bugzilla_snykio_V3"
python -u main.py --dataset ${project} --input_dir \
/scr/dlvp_local_data/reveal_own/ggnn_input/${project}/${project}-original \
--feature_size 225 --model_type ggnn --train 2>&1 | tee logs/$(date "+%m.%d-%H.%M.%S").log