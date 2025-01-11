EXPNAME='extra_MLP_BERT_ray'

RAY_DEDUP_LOGS=0 nohup python3 -u search.py \
    --config_path "../config/MLP_BERT_extra_ray_config.yml" \
    --expName $EXPNAME \
    1> "../log/"$EXPNAME".log" \
    2> "../log/"$EXPNAME".err" &
