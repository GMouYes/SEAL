EXPNAME='extra_MLP_BERT_ray'

CUDA_VISIBLE_DEVICES=0 nohup python3 -u -W ignore tester.py \
    --config_path "../config/MLP_BERT_extra_ray_config.yml" \
    --expName $EXPNAME \
    1> "../log/"$EXPNAME"_infer.log" \
    2> "../log/"$EXPNAME"_infer.err" &

