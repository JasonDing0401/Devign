project="CWE-200"
python -u main.py --dataset ${project} --input_dir \
/scr/dlvp_local_data/reveal_own/ggnn_input/bugzilla_snykio_V3/bugzilla_snykio_V3-original/cwes \
--feature_size 225 --model_type ggnn --model_dataset chrome_debian 2>&1 | tee logs/${project}_$(date "+%m.%d-%H.%M.%S").log