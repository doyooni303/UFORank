import json
import logging
import os
import datetime
import random


import numpy as np
import torch


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logger(log_path: str):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    
    import os
    import time
    os.environ['TZ']='Asia/Seoul'
    time.tzset()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s:%(levelname)s || %(message)s",
                datefmt='%y%m%d-%H시-%M분'
                ),

        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


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

def print_run_time(func):
    import time

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print(
            "Current function [%s] run time is %.8f (s)"
            % (func.__name__, time.time() - local_time)
        )
        return res

    return wrapper


def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))



def plot_setting():

    # Step 3. 셀 실행 후 런타임 재시작
    # Step 4. 라이브러리 호출
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Step 5. 시각화 설정
    import seaborn as sns

    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.set_style("white")

    # Step 6. Linux 한글 사용 설정
    plt.rcParams["font.family"] = ["NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # Step 7. 한글 설정 테스트
    plt.title("한글 테스트")
    plt.show()


def tf_idf(word: str = None, document: list = None):
    n = len(document)  # 문장의 수
    tf, df = 0, 0
    for sentence in document:
        sentence = " ".join(sentence)
        df += int(word in sentence)
        tf += sentence.count(word)
    idf = np.log((n / 1 + df))

    return tf * idf


def softmax(values):
    return np.array([np.exp(v) / np.sum(np.exp(values)) for v in values])


def print_run_time(func):
    import time

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print(
            "Current function [%s] run time is %.8f (s)"
            % (func.__name__, time.time() - local_time)
        )
        return res

    return wrapper


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

def set_logger(log_path: str):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)