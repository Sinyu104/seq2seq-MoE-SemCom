from datasets import Dataset as DS
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
from collections import defaultdict
import pandas as pd
import os


class MultiRC(Dataset):
    def __init__(self, train=True):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading MultiRC dataset")
        
        if train:
            train_file_path = os.path.join('MultiRC', 'train.jsonl')
            self.multirc = pd.read_json(train_file_path, lines=True)
            print(self.multirc)
            
        else:
            test_file_path = os.path.join('MultiRC', 'test.jsonl')
            self.multirc = pd.read_json(test_file_path, lines=True)

aa = MultiRC()
