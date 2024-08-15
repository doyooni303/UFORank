import os
import yaml
import logging
import argparse

from utils import save_json, set_logger

def set_save_path(cfg,date):
    phrase = cfg["MODEL"]["embedding"]["phrase"]
    unit=cfg['MODEL']['glow']["unit"] if cfg["MODEL"]['glow']["unit"] else "None"
    num_clusters = cfg["CLUSTER"]["num_clusters"]
    eps=cfg["CLUSTER"]["dbscan"]["eps"]
    pos=cfg['SCORES']['position']
    similarity=cfg['SCORES']['similarity']
    topic=cfg['SCORES']['topic']
    topic_embedding = cfg['TOPIC']['embedding']
    
    folder_name = f'unit_{unit}_phrase_{phrase}_topicEmbed_{topic_embedding}_num_clusters_{num_clusters}_eps_{eps}_pos_{pos}_sim_{similarity}_topic_{topic}'
    
    cfg["FOLDER_NAME"] = folder_name

    if cfg["MODEL"]["glow"]["pooling"]!= None:
        cfg["SAVEDIR"] = os.path.join(
        cfg["DATA"]["file_path"],
        f"results_{date}",
        cfg["MODEL"]["model"],
        cfg["MODEL"]["glow"]["usage"],
        cfg["MODEL"]["glow"]["pooling"],
        cfg["MODEL"]["glow"]["pretrained"],
        folder_name,)
    else:
        cfg["SAVEDIR"] = os.path.join(
        cfg["DATA"]["file_path"],
        f"results_{date}",
        cfg["MODEL"]["model"],
        cfg["MODEL"]["glow"]["usage"],
        cfg["MODEL"]["glow"]["pretrained"],
        folder_name,)
        
    os.makedirs(cfg["SAVEDIR"], exist_ok=True)
    set_logger(os.path.join(cfg["SAVEDIR"], "logging.log"))
    return cfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date of experiment")
    parser.add_argument("--file_path", type=str, help="name of datset")
    parser.add_argument(
        "--yaml_config", type=str, default="config/main.yaml", help="config filename"
    )
    parser.add_argument(
        "--pretrained", type=str, default="lr_0.001_epochs_20", help="pretrained model"
    )
    parser.add_argument(
        "--unit", type=str, default=None, help="{sentence, document}"
    )
    parser.add_argument(
        "--pooling", type=str, default=None, help="type of pretrained model"
    )
    parser.add_argument("--doc", type=str, help="The way of getting a Document embedding")
    parser.add_argument("--phrase", type=str, help="The way of getting a phrase embedding")
    parser.add_argument("--pos_method", type=str, default='first', help="The way of getting a position weight")
    parser.add_argument("--w1", type=float, default=100, help="Weight of w1")
    parser.add_argument("--w2", type=float, default=1, help="Weight of w2")
    parser.add_argument("--num_clusters", type=float,default=0.2, help="The ratio of the number of clusters")
    parser.add_argument("--clustering", type=str,default='dbscan', help="Clustering Method")
    parser.add_argument("--eps", type=float, help="epsilon for clustering")
    parser.add_argument("--topic_embedding", type=str, default='mean', help="The way of getting a topic embedding vector")
    parser.add_argument("--topic_method", type=str, default='roc', help="The way of getting a topic importance")
    parser.add_argument("--lambda_", type=float, default=0.5, help="Lambda value of MMR")

    parser.add_argument(
        "--similarity", type=str, default=None, help="Apply method on similarity"
    )
    parser.add_argument(
        "--position", type=str, default=None, help="Apply method on position-weight"
    )
    parser.add_argument(
        "--topic", type=str, default=None, help="Apply method on topic-importance"
    )
    args = parser.parse_args()

    # configuration setting
    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)
    
    # date
    cfg["DATE"]=args.date

    # file path
    cfg["DATA"]["file_path"] = args.file_path

    # embedding 종류
    cfg['MODEL']['embedding']['document']=args.doc
    cfg['MODEL']['embedding']['phrase']=args.phrase
    
    # GLOW 활용 여부
    if cfg["MODEL"]["glow"]["usage"] == "with":
        cfg["MODEL"]["glow"]["pretrained"] = args.pretrained
        if args.pooling == None:
            raise NotImplementedError(f"Pooling should be designated")
    cfg["MODEL"]["glow"]["unit"] = args.unit
    cfg["MODEL"]["glow"]["pooling"] = args.pooling
    
    # Position weight
    cfg['POSITION']['method'] = args.pos_method
    cfg['POSITION']['w1'] = args.w1
    cfg['POSITION']['w2'] = args.w2

    # Clsutering
    cfg["CLUSTER"]["clustering"] = args.clustering
    cfg["CLUSTER"]["dbscan"]["eps"] = args.eps
    cfg["CLUSTER"]["num_clusters"] = args.num_clusters
    

    # Clustering importance
    cfg['TOPIC']['embedding'] = args.topic_embedding
    cfg['TOPIC']['method'] = args.topic_method
    cfg['TOPIC']['lambda'] = args.lambda_
    

    # SCORE 종류 여부
    cfg['SCORES']={'topic':args.topic,
                   'position':args.position,
                   'similarity':args.similarity}
    
    
    # SAVE DIR 설정
    cfg = set_save_path(cfg, args.date)
    yaml.dump(cfg, open(args.yaml_config,'w'))

    # save parameters
    save_json(cfg["SAVEDIR"], "basic_arguments.json", cfg)
    set_logger(os.path.join(cfg["SAVEDIR"],'logging.log'))
    logging.info(f"Saving Directory:{cfg['SAVEDIR']}")
