import os
import yaml
import logging
import pickle,json
import yaml
import argparse
from collections import defaultdict
from tqdm import tqdm


import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

from utils import set_logger


def get_clusters(document_feats: list, 
                 cfg):
    """
    Input:
    document_feats(list): A list consisting of each document features in a Dictionary
    clustering_config(dict): A configuration of Agglomerative Clustering
    clustering_name(str): Name of clustering
    Output: cluster_feats(list): A list consisting of cluster features of each document
    ex) cluster_feats[i].keys(): dict_keys(['id', 'cluster_document_feats'])
        cluster_feats[i]['cluster_document_feats']:dict_keys([0, 1, 2, 4, 3]) # cluster number
        cluster_feats[i]['cluster_document_feats']:dict_keys(['candidate_phrases', 'candidate_phrases_embeddings', 'topic_embedding'])
    """
    clustering_name = cfg['CLUSTER']['clustering']
    clustering_config = cfg["CLUSTER"][clustering_name]
    
    topic_embedding = cfg["TOPIC"]['embedding']

    cluster_feats = []
    n_clusters_list = []
    clustering_results = defaultdict(dict)
    for i, feat in tqdm(
        enumerate(document_feats), desc="Getting Clusters", total=len(document_feats)
    ):
        temp = dict()
        id_ = feat["document_id"]
        cp_embeddings = feat["candidate_phrase_embeddings"]

        if clustering_name == "dbscan":
            clustering = DBSCAN(**clustering_config).fit(cp_embeddings)
        else:
            num_clusters = cfg['CLUSTER']['num_clusters']
            n_clusters = max([int(num_clusters*len(feat["candidate_phrases"])),2])
            clustering_config.update({'n_clusters':n_clusters})

            if clustering_name == "agglomerative":
                clustering = AgglomerativeClustering(**clustering_config).fit(cp_embeddings)
            elif clustering_name == "kmeans":
                clustering = KMeans(**clustering_config).fit(cp_embeddings)

        n_clusters_list.append(len(set(clustering.labels_)))
        
        cluster_document_feats = dict()
        
        # phrase 개수, embedding개수, cluster.labels_ 길이 모두 같아야 함
        assert len(feat["candidate_phrases"])==len(cp_embeddings)==len(clustering.labels_)
        
        # dbscan 전용
        if clustering_name == 'dbscan':
            for phrase, embedding, label in zip(
                feat['candidate_phrases'], clustering.components_, clustering.labels_
            ):
                # embedding : np.array / label: int
                if label not in cluster_document_feats.keys():
                    cluster_dict=defaultdict(list)
                    cluster_dict["candidate_phrases"].append(phrase)
                    cluster_dict["candidate_phrase_embeddings"].append(embedding)
                    cluster_document_feats[label] = cluster_dict
                else:
                    cluster_document_feats[label]["candidate_phrases"].append(phrase)
                    cluster_document_feats[label]["candidate_phrase_embeddings"].append(embedding)
        
        else:
            for phrase, embedding, label in zip(
                feat['candidate_phrases'], cp_embeddings, clustering.labels_
            ):
              # embedding : np.array / label: int
                if label not in cluster_document_feats.keys():
                    cluster_dict=defaultdict(list)
                    cluster_dict["candidate_phrases"].append(phrase)
                    cluster_dict["candidate_phrase_embeddings"].append(embedding)
                    cluster_document_feats[label] = cluster_dict
                else:
                    cluster_document_feats[label]["candidate_phrases"].append(phrase)
                    cluster_document_feats[label]["candidate_phrase_embeddings"].append(embedding)  

        # Get Topic embedding
        for label in cluster_document_feats.keys(): 
            embeddings = cluster_document_feats[label]["candidate_phrase_embeddings"]
            if topic_embedding == 'max':
                cluster_document_feats[label]["topic_embedding"] = np.max(
                    embeddings, axis=0
                )
            elif topic_embedding=='mean':
                cluster_document_feats[label]["topic_embedding"] = np.mean(
                    embeddings, axis=0
                )
            elif topic_embedding=='median':
                cluster_document_feats[label]["topic_embedding"] = np.median(
                    embeddings, axis=0
                )

            clustering_results[id_].update({str(label):cluster_document_feats[label]["candidate_phrases"]})
        
        # cluster feats
        temp["id"] = id_
        temp["cluster_document_feats"] = cluster_document_feats
        cluster_feats.append(temp)
    
    logging.info(f"Avg Number of clusters per document:{np.mean(n_clusters_list)}")

    return cluster_feats, clustering_results


if __name__ == "__main__":
    # configuration setting
    parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", type=str, help="name of datset")
    parser.add_argument(
        "--yaml_config", type=str, default="config/main.yaml", help="config filename"
    )

    args = parser.parse_args()

    # configuration setting
    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)
    set_logger(os.path.join(cfg["SAVEDIR"],'logging.log'))

    if os.path.exists(os.path.join(cfg["SAVEDIR"], "cluster_feats.pkl")):
        logging.info(f"clusters_feats.pkl already exists!")

    else:
        document_feats = pickle.load(
            open(
                os.path.join(cfg["SAVEDIR"], f"document_feats_embedding.pkl"),
                "rb",
            )
        )

        cluster_feats, clustering_results = get_clusters(document_feats,cfg)
                                                         

        pickle.dump(
            cluster_feats,
            open(
                os.path.join(cfg["SAVEDIR"], f"cluster_feats.pkl"),
                "wb",
            ),
        )

        json.dump(
            clustering_results,
            open(os.path.join(cfg["SAVEDIR"], f"cluster_results.json"),
                 "w")
        )
        logging.info("Finished making clusters")
