import os, sys
from tqdm import tqdm
import pickle
import argparse
import logging
import yaml

import numpy as np

from collections import defaultdict
from utils import flat_list, set_logger


def weight_function(w1, w2, x, doc_length, threshold=0.5):
    x = (x + 1) / doc_length

    if x <= threshold:
        weight = w1 * (x - threshold) ** (2) + 1
    else:
        weight = w2 * (x - threshold) ** (2) + 1

    return weight


def position_weight(cfg, document_feats, cluster_feats_importance):
    for i in tqdm(range(len(document_feats)),
                  desc='Getting position-biased weights',
                  leave=False):
        
        tokens = document_feats[i]['tokens'] # i-th document
        text = flat_list(tokens)
        doc_length = len(text)
        w1, w2 = cfg["POSITION"]["w1"], cfg["POSITION"]["w2"]
        position_weight_dict = dict()

        for label in cluster_feats_importance[i]['cluster_document_feats'].keys():
            cps = cluster_feats_importance[i]['cluster_document_feats'][label]["candidate_phrases"]
            phrase_weight_dict = defaultdict(list)
            for cp in cps:
                cp_list = cp.split(" ")
                for l in range(doc_length):
                    if cp_list == text[l : l + len(cp_list)]:
                        phrase_weight_dict[cp].append(weight_function(w1, w2, l, doc_length))
            for cp, weight_list in phrase_weight_dict.items():
                if cfg['POSITION']['method'] == 'mean':
                    position_weight_dict.update({cp:np.mean(weight_list)})
                elif cfg['POSITION']['method'] == 'first':
                    position_weight_dict.update({cp:weight_list[0]})
                elif cfg['POSITION']['method'] == 'sum':
                    position_weight_dict.update({cp:np.sum(weight_list)})

            cluster_feats_importance[i]['cluster_document_feats'][label].update({'position_biased_weights':position_weight_dict})

    return cluster_feats_importance


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

    if os.path.exists(os.path.join(cfg["SAVEDIR"], "cluster_feats_positions.pkl")):
        logging.info(f"cluster_feats_positions.pkl already exists!")

    else:
        document_feats = pickle.load(
            open(os.path.join(cfg["SAVEDIR"], f"document_feats_embedding.pkl"), "rb")
        )

        cluster_feats_importance = pickle.load(
            open(os.path.join(cfg["SAVEDIR"], f"cluster_feats_importance.pkl"), "rb")
        )

        document_feats_position = position_weight(cfg, document_feats, cluster_feats_importance)
        pickle.dump(
            document_feats_position,
            open(
                os.path.join(cfg["SAVEDIR"], f"cluster_feats_positions.pkl"),
                "wb",
            ),
        )

        logging.info("Finished getting position-biased weights")