import os, sys
import pickle
from typing import Dict
from tqdm import tqdm
import logging
import argparse
import yaml

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import softmax, set_logger

# implemented from BERTopic https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_mmr.py


def get_cluster_importance(cfg, document_feats, cluster_feats):
    assert len(document_feats) == len(
        cluster_feats
    ), "The number of documents are not same"

    cluster_feats_importance = []
    for doc_feat, clust_feat in tqdm(
        zip(document_feats, cluster_feats),
        desc="Getting the importance of the topic",
        total=len(document_feats),
    ):
        document_embedding = doc_feat["document_embedding"]

        # num_topics = len()
        topic_embeddings_dict = {label:
            clust_feat["cluster_document_feats"][label]["topic_embedding"].tolist()
            for label in clust_feat["cluster_document_feats"].keys()
        }

        mmr_dict = rank_mmr(cfg, document_embedding, topic_embeddings_dict)
        for key in clust_feat["cluster_document_feats"].keys():  # By cluster number
            clust_feat["cluster_document_feats"][key]["importance"] = mmr_dict[key]
        cluster_feats_importance.append(clust_feat)

    return cluster_feats_importance


def rank_mmr(
    cfg,
    document_embedding: np.ndarray,
    topic_embeddings_dict: dict,
) -> Dict:
    """Calculate Maximal Marginal Relevance (MMR)
    between topic nodes and the document.
    This results in a selection of topics that maximize their within diversity with respect to the document.
    Arguments:
        document_embedding: The document embeddings
        topic_embeddings: The embeddings of topic
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         Dict: MMR score of each topic
    """

    diversity = cfg["TOPIC"]["diversity"]

    # Extract similarity within topics, and between topics and the document
    topic_embeddings = np.array([value for value in topic_embeddings_dict.values()])
    # print(topic_embeddings.shape)
    topic_idxs = list(topic_embeddings_dict.keys())
    num_topics = topic_embeddings.shape[0]
    
    # Topic similarities
    topic_similarity = cosine_similarity(topic_embeddings)

    # Topic & Document similairity
    topic_doc_similarity = cosine_similarity(
        topic_embeddings, np.array(document_embedding).reshape(1, -1)
    ).reshape(
        num_topics,
    )

    # Initialize candidate and chosen topic idxs

    chosen_idxs = [np.argmax(topic_doc_similarity)]
    candidates_idx = [i for i in range(num_topics) if i != chosen_idxs[0]]
    for _ in range(num_topics - 1):

        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        target_similarity = -100
        max_mmr = -100
        for j in candidates_idx:
            candidate_similarity = topic_doc_similarity[j]
            for i in chosen_idxs:
                if target_similarity < topic_similarity[i][j]:
                    target_similarity = topic_similarity[i][j]
            temp_mmr = (
                1 - diversity
            ) * candidate_similarity - diversity * target_similarity
            if max_mmr < temp_mmr:
                max_mmr = temp_mmr
                max_mmr_idx = j
                
        # Update mmr_dict
        chosen_idxs.append(max_mmr_idx)
        candidates_idx.remove(max_mmr_idx)

    # Weight by ranks
    mmr_dict = {}
    for k, idx in enumerate(chosen_idxs):
        key = topic_idxs[idx]
        if cfg["TOPIC"]["method"] == "basic":
            mmr_dict[key] = (k + 1) ** (-1)

        elif cfg["TOPIC"]["method"] == "softmax":
            rank = [r ** (-1) for r in range(1, num_topics + 1)]
            weights_by_rank = softmax(rank)
            mmr_dict[key] = weights_by_rank[k]

        elif cfg["TOPIC"]["method"] == "roc":
            weight = num_topics ** (-1) * np.sum(
                [n ** (-1) for n in range(k + 1, num_topics + 1)]
            )
            mmr_dict[key] = weight
    return mmr_dict


if __name__ == "__main__":
    # configuration setting
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_config", type=str, default="config/main.yaml", help="config filename"
    )
    
    args = parser.parse_args()

    # configuration setting
    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)
    set_logger(os.path.join(cfg["SAVEDIR"],'logging.log'))
    
    if os.path.exists(os.path.join(cfg["SAVEDIR"], f"cluster_feats_importance.pkl")):
        logging.info(f"cluster_feats_importance.pkl already exists!")
    else:
        document_feats = pickle.load(
            open(os.path.join(cfg["SAVEDIR"], f"document_feats_embedding.pkl"), "rb")
        )

        cluster_feats = pickle.load(
            open(
                os.path.join(cfg["SAVEDIR"], f"cluster_feats.pkl"),
                "rb",
            )
        )

        cluster_feats_importance = get_cluster_importance(
            cfg, document_feats, cluster_feats
        )

        pickle.dump(
            cluster_feats_importance,
            open(
                os.path.join(cfg["SAVEDIR"], f"cluster_feats_importance.pkl"),
                "wb",
            ),
        )

        logging.info("Finished making clusters importance")
