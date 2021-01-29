#!/bin/bash

# Note these configurations: 
# - shared src tgt vocabulary
# - language embedding on decoder input (to force correct language), addtive or concatenative
# - Tgt language token replaces normal BOS token

input=$1
name=$2

size=512
if [ $# -ne 2 ]; then
    size=$3
fi
innersize=$((size*4))

if [ -z $LAYER ]; then
    LAYER=8
fi

if [ -z $TRANSFORMER ]; then
    TRANSFORMER=transformer
fi

if [ -z "$BASEDIR" ]; then
    BASEDIR=/
fi

if [ -z "$NMTDIR" ]; then
    NMTDIR=/opt/NMTGMinor/
fi

if [ -z "$GPU" ]; then
    GPU=0
fi

if [ $GPU -eq -1 ]; then
    gpu_string_train=""
    gpu_string_avg=""
else
    gpu_string_train="-gpus "$GPU
    gpu_string_avg="-gpu "$GPU
fi

if [ ! -z "$FP16" ]; then
    gpu_string_train=$gpu_string_train" -fp16 -fp16_mixed"
fi

echo 'GPU parameters: '$gpu_string_train

if [ -z $OPTIM ]; then
    optim_str="-optim adam -update_method noam"
elif [ $OPTIM == "noam" ]; then
    optim_str="-optim adam -update_method noam"
elif [ $OPTIM == "adam" ]; then
    optim_str="-optim adam"
else 
    echo "Unkown optim methods "$OPTIM
    exit;
fi

if [ -z "$LR" ]; then
    LR=2
fi

if [ -z "$WUS" ]; then
    WUS=8000
fi

if [ -z "$EPOCHS" ]; then 
    EPOCHS=128
fi

if [ -z "$HEAD" ]; then
    HEAD=8
fi

if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=3584
fi

if [ -z "$INPUT_TYPE" ]; then
    INPUT_TYPE=word
fi

if [ -z "$SKIP_TRAIN" ]; then 
    SKIP_TRAIN=false
fi

if [ -z "$MULTILAN" ]; then
    MULTILAN=false
fi

if [ "$LAN_EMB" == true ]; then
    magic_str=$magic_str" -use_language_embedding"
fi

if [ "$LAN_EMB_CONCAT" == true ]; then
    magic_str=$magic_str" -language_embedding_type concat"
fi

if [ -z "$SEED" ]; then
    SEED=8877
fi

if [ -z "$DEATH" ]; then
    DEATH=0.0
fi

python3 -u $NMTDIR/train.py  -data $BASEDIR/model/${name}/train -data_format bin \
       -save_model $BASEDIR/model/${name}/checkpoints/model \
       -model $TRANSFORMER \
       -batch_size_words $BATCH_SIZE \
       -batch_size_update 24568 \
       -batch_size_sents 9999 \
       -batch_size_multiplier 8 \
       -checkpointing 0 \
       -layers $LAYER \
       -encoder_layers $ENC_LAYER \
       -model_size $size \
       -inner_size $innersize \
       -n_heads $HEAD \
       -dropout 0.2 \
       -attn_dropout 0.2 \
       -word_dropout 0.1 \
       -emb_dropout 0.2 \
       -label_smoothing 0.1 \
       -epochs $EPOCHS \
       $optim_str \
       -learning_rate $LR \
       -normalize_gradient \
       -warmup_steps $WUS \
       -tie_weights \
       -seed $SEED \
       -log_interval 1000 \
       -death_rate $DEATH \
       -join_embedding \
       -data_format mmem \
       -update_frequency -1 \
       $magic_str $gpu_string_train &> $BASEDIR/model/${name}/train.log

checkpoints=""

for f in `ls $BASEDIR/model/${name}/checkpoints/model_ppl_*`
do
    checkpoints=$checkpoints"${f}|"
done
checkpoints=`echo $checkpoints | sed -e "s/|$//g"`

python3 -u $NMTDIR/average_checkpoints.py $gpu_string_avg \
					-models $checkpoints \
					-output $BASEDIR/model/${name}/model.pt

rm -r $BASEDIR/tmp/${name}/
