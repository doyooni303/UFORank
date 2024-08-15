DATE="240815"
YAML_CONFIG='config/longformer_with_pretrained.yaml'
FILE_PATH=data/nus
DOC=max
PHRASE=max
POS_METHOD=sum
W1=100
W2=1
UNIT=sentence
POOLING=first-last-avg
PRETRAINED='lr_0.001_epochs_5'
CLUSTERING='kmeans' #'agglomerative' #'kmeans' # 'dbscan' # 
EPS=00
NUM_CLUSTERS=0.5
TOPIC_EMBEDDING=mean
python3 src/save_params.py --file_path $FILE_PATH \
                            --date $DATE \
                            --unit $UNIT \
                            --pooling $POOLING \
                            --yaml_config $YAML_CONFIG \
                            --pretrained $PRETRAINED \
                            --doc $DOC \
                            --phrase $PHRASE \
                            --pos_method $POS_METHOD \
                            --clustering $CLUSTERING \
                            --num_clusters $NUM_CLUSTERS \
                            --eps $EPS \
                            --topic_embedding $TOPIC_EMBEDDING \
                            --w1 $W1 \
                            --w2 $W2 \
                            
python3 src/get_embedding.py \
    --yaml_config $YAML_CONFIG \

python3 src/get_clusters.py \
    --yaml_config $YAML_CONFIG \

python3 src/get_cluster_importance.py \
    --yaml_config $YAML_CONFIG \

python3 src/get_position_biased_weights.py \
        --yaml_config $YAML_CONFIG \

python3 src/get_scores.py \
    --yaml_config $YAML_CONFIG \

rm -r ${FILE_PATH}/results_${DATE}/longformer/*/*/*/*/*.pkl