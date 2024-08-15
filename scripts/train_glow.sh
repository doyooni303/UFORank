export LC_ALL=C.UTF-8
export LANG=C.UTF-8
FILE_NAME='test.json'
MODEL='longformer' 
MODEL_NAME='allenai/longformer-base-4096'
DEVICE0=0
DEVICE1=1
MAX_LENGTH=4098
ACCUM_STEPS=1
FILE_PATH='data/SemEval2010'
UNIT=document
POOLING=mean
EPOCHS=5
LR=1e-3
python3 bertflow/train_bertflow.py\
    --file_path $FILE_PATH\
    --file_name $FILE_NAME\
    --model $MODEL\
    --model_name $MODEL_NAME\
    --unit $UNIT\
    --pooling $POOLING\
    --lr $LR\
    --device0 $DEVICE0\
    --device1 $DEVICE1\
    --epochs $EPOCHS\
    --accum_steps $ACCUM_STEPS\
    --max_length $MAX_LENGTH\