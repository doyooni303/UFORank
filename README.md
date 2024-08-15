# UFORank
Official repository of **UFORank: Unified Framework of Unsupervised Keyphrase Extraction for Long Documents**. Our code is based on the following links:
- [ Unsupervised Keyphrase Extraction by Jointly Modeling Local and Global Context(Liang et al., 2021)](https://github.com/xnliang98/uke_ccrank)
- [MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction(Zhang et al., 2021)](https://github.com/LinhanZ/mderank)
- [PromptRank: Unsupervised Keyphrase Extraction using Prompt(Kong et al., 2023)](https://github.com/NKU-HLT/PromptRank/blob/master/README.md).

## Environment

```
tabulate
tqdm
wandb (if you need)
numpy 1.22.4
nltk 3.7
spacy 3.7.2
torch 1.12.1
transformers 4.21.0
```

## Pre-requisites
Make sure to set your path on `UFORank` folder to run all the script files.

To get pretrained Glow weights, you should run `train_glow.sh` in the scripts folder. 
```bash
bash scripts/train_glow.sh
```
Then, it will be saved in a folder, `bertflow_model`, in each dataset folder.

## Run
* If you have the pretrained Glow weights,
```bash
bash scripts/get_main_results.sh
```

* If you don't want to utilize Glow,
```bash
bash scripts/get_main_without_glow_results.sh
```

## Hyper parameters
Hyper parameters on each dataset as mentioned in the paper are like the followings,

|  Dataset | Topic Embedding |  Glow Pooling  | Glow Epochs |  Glow LR | Method | cluster Number Ratio |
|:--------:|:---------------:|:--------------:|:-----------:|:--------:|:------:|:--------------------:|
|  SemEval |       mean      |      mean      |      5      | 1E-3 | kmeans |          0.6         |
|    NUS   |       mean      | first-last-avg |      5      | 1E-3 | kmeans |          0.5         |
| Krapivin |       mean      |      mean      |      1      | 1E-3 | kmeans |          0.6         |

## Performance
| **SemEval2010** 	|           	|           	|         	|  **NUS** 	|           	|           	|         	| **Krapivin** 	|           	|           	|         	|
|:---------------:	|:---------:	|:---------:	|:-------:	|:--------:	|:---------:	|:---------:	|:-------:	|:------------:	|:---------:	|:---------:	|:-------:	|
|     **Top5**    	| **Top10** 	| **Top15** 	| **AVG** 	| **Top5** 	| **Top10** 	| **Top15** 	| **AVG** 	|   **Top5**   	| **Top10** 	| **Top15** 	| **AVG** 	|
|      13.83      	|   18.95   	|   20.30   	|  17.69  	|   19.34  	|   22.30   	|   21.76   	|  21.13  	|     13.37    	|   13.75   	|   12.90   	|  13.34  	|
