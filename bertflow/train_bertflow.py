# encoding: utf-8
from tqdm import tqdm
import os
import json
import pdb
import argparse

import nltk
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import LongformerTokenizer, BertTokenizer
from nltk.corpus import stopwords
import wandb

from tflow_utils import TransformerGlow, AdamWeightDecayOptimizer

nltk.download("stopwords")
stopword_dict = set(stopwords.words("english"))


def save_json(path: str, file_name: str, dictionary: dict):
    """Saves dict of floats in json file

    Args:
        path: Folder name you wish to save in
        file_name: The name of file that will be saved as .json
        dictionary: Dictionary you want to save
    """

    PATH = os.path.join(path, file_name)
    if not os.path.exists(path):
        print("Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)
    else:
        print("Directory exists! ")

    with open(PATH, "w", encoding="utf-8") as make_file:
        json.dump(dictionary, make_file, ensure_ascii=False, indent=4)


def get_filepath_list(path):
    g = os.walk(path)
    file_path_list = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_path_list.append(os.path.join(path, file_name))
    return file_path_list


def read_json_by_line(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]

def flat_list(l):
    return [x for ll in l for x in ll]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, default="data/DUC2001", help="data dir"
    )
    parser.add_argument(
        "--file_name", type=str, default="test.json", help="data name with json format"
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model", type=str, default="BERT")
    parser.add_argument("--unit", type=str, default="sentence")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device0", type=int, default=0)
    parser.add_argument("--device1", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--autocast", type=bool, default=True)
    args = parser.parse_args()

    # save path
    name = f"lr_{args.lr}_epochs_{args.epochs}"
    results_path = os.path.join(args.file_path, "bertflow_model",args.model, args.unit,args.pooling)
    os.makedirs(results_path, exist_ok=True)

    # wandb setting
    wandb.init(
        project="bertflow training",
        name=name,
        reinit=True,
        group=args.file_path,
    )
    wandb.config.update(args)

    # model and device settings
    print(f"# of GPUs:{torch.cuda.device_count()}")
    if torch.cuda.device_count() >= 2:
        device0 = torch.device(args.device0)
        device1 = torch.device(args.device1)

    elif torch.cuda.device_count() == 1:
        device0 = torch.device(args.device0)
        device1 = None
    else:
        device0 = torch.device("cpu")
    bertflow = TransformerGlow(
        device0=device0,
        device1=device1,
        model_name_or_path=args.model_name,
        pooling=args.pooling,
    )  # pooling could be 'mean', 'max', 'cls' or 'first-last-avg' (mean pooling over the first and the last layers)
    # bertflow.to(device)

    mem_params = sum(
        [param.nelement() * param.element_size() for param in bertflow.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in bertflow.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(f"model size(memory): {mem} bytes")

    if args.model == "BERT":
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    elif args.model == "longformer":
        tokenizer = LongformerTokenizer.from_pretrained(args.model_name)

    # optimizer and scheduler settings
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in bertflow.glow.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in bertflow.glow.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWeightDecayOptimizer(
        params=optimizer_grouped_parameters,
        lr=args.lr,
        eps=1e-8,
    )

    # data reading
    json_list = read_json_by_line(os.path.join(args.file_path, args.file_name))

    # amp
    scaler = GradScaler()

    # training
    bertflow.train()
    step = 0
    whole_batch,mini_batch = 0,0
    for epoch in tqdm(range(args.epochs), desc="Epochs", total=args.epochs):
        doc_loss = 0
        for i, doc_feat in tqdm(
            enumerate(json_list), desc="Num of Iters", total=len(json_list),leave=False
        ):
            if args.unit == "document":
                try:
                    optimizer.zero_grad()
                    model_inputs = tokenizer(
                        doc_feat["tokens"],
                        add_special_tokens=True,
                        return_tensors="pt",
                        max_length=args.max_length,
                        padding="longest",
                        truncation=True,
                        is_split_into_words=True,
                    ).to(device0)

                    if args.autocast:
                        with autocast(dtype=torch.float16):
                            z, loss = bertflow(
                                input_ids[0],
                                attention_mask[0],
                                return_loss=True,
                            )  # Here z is the sentence embedding
                    else:
                        z, loss = bertflow(
                            model_inputs["input_ids"],
                            model_inputs["attention_mask"],
                            return_loss=True,
                        )

                    scaler.scale(loss).backward()
                    doc_loss+=loss
                    if (i + 1) % args.accum_steps == 0:
                        # optimizer.step()
                        scaler.step(optimizer)
                        scaler.update()
                        wandb.log({"loss": doc_loss/args.accum_steps, "epoch": epoch, "steps": step})
                        doc_loss = 0
                        step += 1

                    torch.cuda.empty_cache()
                    whole_batch+=1
                except:
                    batch_loss = 0
                    optimizer.zero_grad()
                    model_inputs = tokenizer(
                        doc_feat["tokens"],
                        add_special_tokens=True,
                        return_tensors="pt",
                        max_length=args.max_length,
                        padding="longest",
                        truncation=True,
                        is_split_into_words=True,
                    ).to(device0)

                    input_ids_dataset = TensorDataset(model_inputs['input_ids'])
                    attention_mask_dataset  = TensorDataset(model_inputs['attention_mask'])

                    input_ids_loader = DataLoader(input_ids_dataset, batch_size = 16, shuffle=False, drop_last = False)
                    attention_mask_loader = DataLoader(attention_mask_dataset, batch_size = 16, shuffle=False, drop_last = False)
                    j=0
                    for input_ids, attention_mask in tqdm(zip(input_ids_loader,attention_mask_loader), 
                                                        desc = f"Batches in a document",
                                                        total = len(input_ids_loader),
                                                        leave=False,
                                                        ):
                        if args.autocast:
                            with autocast(dtype=torch.float16):
                                z, loss = bertflow(
                                    input_ids[0],
                                    attention_mask[0],
                                    return_loss=True,
                                )  # Here z is the sentence embedding
                        else:
                            z, loss = bertflow(
                                model_inputs["input_ids"],
                                model_inputs["attention_mask"],
                                return_loss=True,
                            )

                        doc_loss += loss

                        scaler.scale(loss).backward()

                        if (j + 1) % args.accum_steps == 0:
                            # optimizer.step()
                            scaler.step(optimizer)
                            scaler.update()

                            wandb.log({"loss": doc_loss/args.accum_steps, "epoch": epoch, "steps": step})
                            step += 1
                        j+=1

                    torch.cuda.empty_cache()
                    mini_batch+=1
            elif args.unit == 'sentence': 
                batch_loss = 0
                optimizer.zero_grad()
                model_inputs = tokenizer(
                    doc_feat["tokens"],
                    add_special_tokens=True,
                    return_tensors="pt",
                    max_length=args.max_length,
                    padding="longest",
                    truncation=True,
                    is_split_into_words=True,
                ).to(device0)

                input_ids_dataset = TensorDataset(model_inputs['input_ids'])
                attention_mask_dataset  = TensorDataset(model_inputs['attention_mask'])

                input_ids_loader = DataLoader(input_ids_dataset, batch_size = 16, shuffle=False, drop_last = False)
                attention_mask_loader = DataLoader(attention_mask_dataset, batch_size = 16, shuffle=False, drop_last = False)
                j=0
                for input_ids, attention_mask in tqdm(zip(input_ids_loader,attention_mask_loader), 
                                                    desc = f"Batches in a document",
                                                    total = len(input_ids_loader),
                                                    leave=False,
                                                    ):
                    if args.autocast:
                        with autocast(dtype=torch.float16):
                            z, loss = bertflow(
                                input_ids[0],
                                attention_mask[0],
                                return_loss=True,
                            )  # Here z is the sentence embedding
                    else:
                        z, loss = bertflow(
                            model_inputs["input_ids"],
                            model_inputs["attention_mask"],
                            return_loss=True,
                        )

                    doc_loss += loss

                    scaler.scale(loss).backward()

                    if (j + 1) % args.accum_steps == 0:
                        # optimizer.step()
                        scaler.step(optimizer)
                        scaler.update()

                        wandb.log({"loss": doc_loss/args.accum_steps, "epoch": epoch, "steps": step})
                        step += 1
                    j+=1

                torch.cuda.empty_cache()
                mini_batch+=1
    
    if args.unit == 'document':
        print(f"Train counts|whole batch:{whole_batch}|mini batch:{mini_batch}")

    wandb.finish()


    output_path = os.path.join(results_path, name)
    bertflow.save_pretrained(output_path)  # Save model

    # save args
    save_json(output_path, "args.json", vars(args))
