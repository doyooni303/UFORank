import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import logging
import argparse

import yaml
import nltk
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity

from utils import save_json, set_logger, print_run_time


def get_exp_log(array, method:str = None)->np.ndarray:
    if method == 'exp':
        array = np.exp(array)
    elif method =='log':
        array = np.log(array)
    
    return array
            
def get_ablation_score(topic_importance,
                       pos_weight,
                       topic_similarity,
                       ablation:str):
    if ablation =="None":
        score = topic_importance * pos_weight * topic_similarity
    elif ablation =="pos_weight":
        score = topic_importance * topic_similarity,
    elif ablation =="topic_importance":
        score = pos_weight * topic_similarity
    elif ablation=="similarity":
        score = topic_importance * pos_weight
    
    return score

def get_final_scores(cfg,
                     cluster_feats_positions: list,
                     ablation: str):
    SCORES = []

    for i in tqdm(
        range(len(cluster_feats_positions)),
        desc="Getting the final scores",
        leave=False,
    ):
        cluster_document_feats = cluster_feats_positions[i]["cluster_document_feats"] # dict: label(cluster number)
        
        final_scores = []

        # cluster 내에서 cp score 계산 해야지
        for label in cluster_document_feats.keys():
            topic_importance = cluster_document_feats[label]["importance"]
            topic_embedding = cluster_document_feats[label]["topic_embedding"]
            cps = cluster_document_feats[label]["candidate_phrases"]
            cp_embeddings = cluster_document_feats[label]["candidate_phrase_embeddings"]
            position_weights_dict = cluster_document_feats[label]['position_biased_weights'] 
            
            for j,(cp, embedding) in enumerate(zip(cps,cp_embeddings)):
                pos_weight = position_weights_dict[cp]
                
                topic_similarity = cosine_similarity(np.array(embedding).reshape(1,-1),
                                                    topic_embedding.reshape(1,-1)
                                                    )[0][0]
            
  
                topic_importance = get_exp_log(topic_importance, cfg['SCORES']['topic'])
                pos_weight = get_exp_log(pos_weight, cfg['SCORES']['position'])
                topic_similarity = get_exp_log(topic_similarity, cfg['SCORES']['similarity'])
                
                score = get_ablation_score(topic_importance,pos_weight,topic_similarity,ablation)
                final_scores.append((cp,score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        SCORES.append(final_scores)

    return  SCORES

def get_topk_candidates(document_feats_positions, SCORES):
    top_k=[5,10,15]
    assert len(document_feats_positions) == len(SCORES)
    F1 = defaultdict(list)
    P = defaultdict(list)
    R = defaultdict(list)
    
    pred_stemmed, predictions= [],[]
    porter = nltk.PorterStemmer()

    for i in tqdm(range(len(SCORES)), desc="Getting the predictions & F1-scores", leave=False,):
        # 하나의 문서에 대해서
        doc_feat_pos = document_feats_positions[i]
        scores = SCORES[i]  # list[(phrase,score)]
        keyphrases = set(doc_feat_pos["keyphrases"])
        candidate_phrases = [x[0].strip() for x in scores]
        
        # 후보 phrase 중복 미허용
        predicted_candidation_stem = []
        for phrase in candidate_phrases:
            phrase = " ".join([porter.stem(x) for x in phrase.strip().split(" ")])
            if phrase not in predicted_candidation_stem:
                predicted_candidation_stem.append(phrase)
        
        predictions.append(candidate_phrases)
        pred_stemmed.append(predicted_candidation_stem)
        p, r = get_precision_recall(
            predicted_candidation_stem, keyphrases, maxDepth=15
        )  # list 반환

        for k in top_k:
            precision = p[k - 1]
            recall = r[k - 1]
            if precision + recall > 0:
                F1[k].append((2 * (precision * recall)) / (precision + recall))
            else:
                F1[k].append(0)
            P[k].append(precision)
            R[k].append(recall)

    return pred_stemmed, predictions, P, R, F1


def get_precision_recall(candidates, references, maxDepth=15):
    precision = []
    recall = []
    for i in range(maxDepth):
        tp = set(candidates[:i+1]).intersection(set(references))
        p=len(tp)/len(set(candidates[:i+1]))
        r=len(tp)/len(set(references))
        precision.append(p)
        recall.append(r)
    return precision, recall

@print_run_time
def main(cfg):
    document_feats_embedding = pickle.load(
        open(
            os.path.join(cfg["SAVEDIR"], f"document_feats_embedding.pkl"),
            "rb",
        )
    )

    cluster_feats_positions = pickle.load(
        open(
            os.path.join(cfg["SAVEDIR"], f"cluster_feats_positions.pkl"),
            "rb",
        )
    )

    # Including ablations
    ablation_results_dict = defaultdict(dict)
    score_dict={}
    for ablation in tqdm(
        ["None", "pos_weight", "topic_importance", "similarity"], desc="Ablation study",leave=False
    ):
        SCORES = get_final_scores(cfg,cluster_feats_positions, ablation)
        score_dict[ablation] = SCORES
        pred_stemmed, predictions,P, R, F1 = get_topk_candidates(document_feats_embedding, SCORES)
        prediction_dict = {}
        for i in range(len(document_feats_embedding)):
            doc_feat = document_feats_embedding[i]
            prediction_stemmed = pred_stemmed[i]
            prediction_dict[doc_feat["document_id"]] = {
                "prediction":predictions[:15],
                "prediction_stemmed": prediction_stemmed[:15],
                "keyphrases": doc_feat["keyphrases"],
            }
        ablation_results_dict[ablation]["prediction"] = prediction_dict
        ablation_results_dict[ablation]["precision"] = P
        ablation_results_dict[ablation]["recall"] = R
        ablation_results_dict[ablation]["f1"] = F1
        
    # save predictions and all the scores
    save_json(cfg["SAVEDIR"], "predictions.json", ablation_results_dict)
    save_json(cfg["SAVEDIR"], "scores.json", score_dict)

    # display and save Top-k socres
    name = "_".join(
            [
                cfg["MODEL"]["model"],
                cfg["MODEL"]["glow"]["usage"],
                cfg["MODEL"]["glow"]["pooling"],
                cfg["MODEL"]["glow"]["pretrained"],
                cfg["FOLDER_NAME"],
            ]
            ) if cfg["MODEL"]["glow"]["pooling"] is not None \
            else "_".join(
            [
                cfg["MODEL"]["model"],
                cfg["MODEL"]["glow"]["usage"],
                "None",
                cfg["MODEL"]["glow"]["pretrained"],
                cfg["FOLDER_NAME"],
            ]
            )
    clustering = cfg["CLUSTER"]["clustering"]
    eps_nclusters = cfg["CLUSTER"][clustering]['eps'] if clustering == 'dbscan' else cfg["CLUSTER"]["num_clusters"]
    default_header = ['Name',
              'Document embedding',
              'Phrase embedding', 
              'Glow pooling',
              'Glow pretrained',
              'Pos method',
              'Topic method',
              'Clustering',
              "EPS/n_lusters"
              ]
    default_value = [name,
                     cfg['MODEL']['embedding']["document"],
                     cfg['MODEL']['embedding']["phrase"],
                     cfg['MODEL']['glow']['pooling'],
                     cfg['MODEL']['glow']['pretrained'],
                     cfg["POSITION"]["method"],
                     cfg["TOPIC"]["method"],
                     cfg["CLUSTER"]["clustering"],
                     eps_nclusters
                     ]

    new_result_df = pd.DataFrame()
    for ablation in ["None", "pos_weight", "topic_importance", "similarity"]:
        result_dict = {}
        F1 = ablation_results_dict[ablation]["f1"]
        result_dict.update({"Ablation":ablation})
        result_dict.update({
            f"Top {i}": round(np.mean(F1[i]) * 100, 2) for i in [5,10,15]
        })
        temp_header = default_header + list(result_dict.keys())
        temp_value = default_value + list(result_dict.values())
        temp_dict = {k:v for k,v in zip(temp_header,temp_value)}
        new_result_df = pd.concat([new_result_df, pd.Series(temp_dict).to_frame().T])
    

    logging.info(
        tabulate(
            new_result_df,
            headers=new_result_df.columns,
            tablefmt="outline",
        )
    )
    

    logging.info("Finished getting the final scores!")


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
    main(cfg)
    
    