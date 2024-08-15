# encoding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle
import logging
import argparse
from typing import Tuple, List

import yaml
import nltk
import torch
import numpy as np

from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertModel,
    LongformerModel,
    LongformerTokenizer,
)
from nltk.corpus import stopwords

from bertflow.tflow_utils import TransformerGlow, Glow
from utils import read_json_by_line, flat_list, set_logger, print_run_time

def encode_sentences(cfg, tokenizer, model, json_list, device)->List[dict]:
    document_encodings = []
    for document in tqdm(json_list, desc="Encoding sentences"):
        document_id = document["document_id"]

        tokens = flat_list(document["tokens"])
        word_emb, cls_token = encode_sentence(cfg, tokenizer, model, tokens, device)

        document_encodings.append(
            {
                "document_id": document_id,
                "doc_cls": cls_token,
                "word_emb": word_emb,
            }
        )

    return document_encodings


def encode_sentence(cfg, tokenizer, model, tokens, device):

    if cfg["MODEL"]["model"] == "BERT":
        is_split = []
        input_tokens = ["[CLS]"]
        for token in tokens:
            tmp = tokenizer.tokenize(token)

            if len(input_tokens) + len(tmp) >= cfg["MODEL"]["max_length"] - 1:
                break
            else:
                input_tokens.extend(tmp)
                is_split.append(len(tmp))
        input_tokens += ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    elif cfg["MODEL"]["model"] == "longformer":
        is_split = []
        input_tokens = ["<s>"]
        for token in tokens:
            tmp = tokenizer.tokenize(token)

            if len(input_tokens) + len(tmp) >= cfg["MODEL"]["max_length"] - 2:
                break
            else:
                input_tokens.extend(tmp)
                is_split.append(len(tmp))
        input_tokens += ["</s>"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    input_ids = torch.LongTensor([input_ids]).to(device)
    outputs = model(input_ids, output_hidden_states=True)
    o1 = outputs.last_hidden_state.squeeze().cpu().detach().numpy()  # token embeddings
    cls_emb = outputs.pooler_output.squeeze().cpu().detach().numpy()  # [CLS],<s> embedding

    tokens_emb = []
    i = 1
    for j in is_split:
        if j == 1:
            tokens_emb.append(o1[i])
            i += 1
        else:
            tokens_emb.append(
                np.array(o1[i : i + j]).max(axis=0) # np.array(o1[i : i + j]).mean(axis=0)
            )  # word embedding = Max-Pooling(token-embeddings)
            i += j

    assert len(tokens_emb) == len(is_split)
    return tokens_emb, cls_emb



def get_document_feats(cfg, json_list, document_encodings, glow, stopword_dict=None, device=None)->List[dict]:
    np.random.seed(33)
    n = np.random.choice(len(json_list))
    document_feats = []
    for k,(document, doc_encoding) in tqdm(
        enumerate(zip(json_list, document_encodings)),
        desc="Getting doc,phrase embeddings",
        total=len(json_list),
    ):
        assert document["document_id"] == doc_encoding["document_id"]

        sentence = flat_list(document["tokens"])
        sentence_pos = flat_list(document["tokens_pos"])
        word_embeddings = doc_encoding["word_emb"]
        tokens_tagged = list(zip(sentence, sentence_pos))
        for i, token in enumerate(sentence):
            if token.lower() in stopword_dict:
                tokens_tagged[i] = (token, "IN")
        total_candidate_phrases = extract_candidates(tokens_tagged)
        

        if glow is not None:
            phrase_embeddings, candidate_phrases = get_phrase_embeddings(
                cfg, total_candidate_phrases, word_embeddings
            )
            document_embedding = get_document_embeddings(
                cfg, doc_encoding["doc_cls"], word_embeddings
            )

            doc_phrase_embeddings = [document_embedding] + phrase_embeddings


            glow_doc_phrase_embeddings, _ = glow(
                torch.Tensor(np.array(doc_phrase_embeddings)).to(device)
            )
            phrase_embeddings = glow_doc_phrase_embeddings.tolist()[1:]
            document_embedding = glow_doc_phrase_embeddings.tolist()[0]

            


        else:  # Only PLM

            phrase_embeddings, candidate_phrases = get_phrase_embeddings(
                cfg, total_candidate_phrases, word_embeddings
            )
            document_embedding = get_document_embeddings(
                cfg, doc_encoding["doc_cls"], word_embeddings
            )

        if k==n:
            logging.info(f"candidate phrases:{candidate_phrases}")
        
        document_feats.append(
            {
                "document_id": document["document_id"],
                "tokens": document["tokens"],
                "candidate_phrases": candidate_phrases, # 미중복
                "candidate_phrase_embeddings": phrase_embeddings,
                "document_embedding": document_embedding,
                "keyphrases": document["keyphrases"],
            }
        )
    return document_feats

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    GRAMMAR1 = """  NP:
            {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

    GRAMMAR2 = """  NP:
            {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

    GRAMMAR3 = """  NP:
            {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    # single_candidate_phrases = []
    total_candidate_phrases = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if isinstance(token, nltk.tree.Tree) and token._label == "NP":
            np = " ".join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            total_candidate_phrases.append((np, start_end))

        else:
            count += 1

    return total_candidate_phrases

def get_phrase_embeddings(cfg, phrases, word_embeddings)->Tuple[List[list], List[str]]:
    # 중복 phrase도 모두 다른 임베딩으로
    embeddings = []
    candidate_phrases = []

    for phrase, (i, j) in phrases:
        if j <= i:
            continue
        if j >= len(word_embeddings):
            break
        if cfg["MODEL"]["embedding"]["phrase"] == "mean":
            embeddings.append(np.mean(word_embeddings[i:j],axis=0))
        elif cfg["MODEL"]["embedding"]["phrase"] == "max":
            embeddings.append(np.max(word_embeddings[i:j],axis=0))
        candidate_phrases.append(phrase)
    
    return embeddings, candidate_phrases

def get_document_embeddings(cfg, cls_embedding: np.array, word_embeddings: list)-> list:
    if cfg["MODEL"]["embedding"]["document"] == "cls":
        document_embedding = cls_embedding

    elif cfg["MODEL"]["embedding"]["document"] == "max":
        document_embedding = np.max(word_embeddings, axis=0)  # Tokens from glow outputs

    elif cfg["MODEL"]["embedding"]["document"] == "mean":
        document_embedding = np.mean(word_embeddings, axis=0)  # Tokens from glow outputs

    return document_embedding.tolist()

@print_run_time
def main(cfg):
    if os.path.exists(os.path.join(cfg["SAVEDIR"], "document_feats_embedding.pkl")):
        logging.info(f"document_feats_embedding.pkl already exists!")

    else:
        # nltk setting
        nltk.download("stopwords")
        stopwords_dict = set(stopwords.words("english"))
        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")

        # Model setting
        if cfg["MODEL"]["model"] == "BERT":
            tokenizer = BertTokenizer.from_pretrained(cfg["MODEL"]["model_name"])
            model = BertModel.from_pretrained(cfg["MODEL"]["model_name"]).to(device)
        elif cfg["MODEL"]["model"] == "longformer":
            tokenizer = LongformerTokenizer.from_pretrained(cfg["MODEL"]["model_name"])
            model = LongformerModel.from_pretrained(cfg["MODEL"]["model_name"]).to(
                device
            )

        # With/without Glow
        if cfg["MODEL"]["glow"]["usage"] == "with":
            if cfg["MODEL"]["glow"]["pretrained"] != "None":
                if cfg["MODEL"]["glow"]['pooling'] != None:
                    pretrained_name = '_'.join([cfg["MODEL"]["model"],cfg["MODEL"]["glow"]['pooling'],cfg["MODEL"]["glow"]["pretrained"]])
                else:
                    pretrained_name = '_'.join([cfg["MODEL"]["model"],cfg["MODEL"]["glow"]["pretrained"]])
                logging.info(
                    f"Loading Pre-trained bertflow:{pretrained_name}"
                )
                pretrained_path = os.path.join(
                    cfg["DATA"]["file_path"],
                    "bertflow_model",
                    cfg["MODEL"]["model"],
                    cfg["MODEL"]["glow"]["unit"],
                    cfg["MODEL"]["glow"]['pooling'],
                    cfg["MODEL"]["glow"]["pretrained"],
                )
                bertflow_model = TransformerGlow.from_pretrained(pretrained_path)
                glow = bertflow_model.glow.to(device)
            else:
                hidden_size = model.config.hidden_size
                glow = Glow(hidden_size).to(device)
        else:
            glow = None

        json_list = read_json_by_line(
            os.path.join(cfg["DATA"]["file_path"], cfg["DATA"]["file_name"])
        )

        document_embeddings = encode_sentences(cfg, tokenizer, model, json_list, device)
        document_feats = get_document_feats(cfg, json_list, document_embeddings, glow,stopwords_dict,device)

        pickle.dump(
            document_feats,
            open(
                os.path.join(cfg["SAVEDIR"], f"document_feats_embedding.pkl"),
                "wb",
            ),
        )
        logging.info(f"Finished saving embeddings!")



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
    