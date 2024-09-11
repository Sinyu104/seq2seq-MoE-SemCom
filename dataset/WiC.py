from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch
import os

def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label == 'T\n':
        return "Yes"
    elif label == 'F\n':
        return "No"
    else:
        raise ValueError("Invalid label")

def set_prompt(idx = 0):
    prompt = [
    """Does the word "{Word}" have the same meaning in these two sentences?\n{Sentence1}\n{Sentence2}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-Yes\n-No """,
    ]
    return options[idx]


class WiC(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading WiC dataset")

        if train:
            dataset_path = os.path.join('dataset', 'WiC_dataset', 'train', 'train.data.txt')
            gold_path = os.path.join('dataset', 'WiC_dataset', 'train', 'train.gold.txt')

        else:
            dataset_path = os.path.join('dataset','WiC_dataset', 'test', 'test.data.txt')
            gold_path = os.path.join('dataset','WiC_dataset', 'test', 'test.gold.txt')
        

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        with open(dataset_path, 'r') as file:
            self.wic = file.readlines()
            
        
        with open(gold_path, 'r') as file:
            self.wic_gold = file.readlines()

        for idx, sample in enumerate(self.wic):
            parts = sample.strip().split('\t')
            input_text = prompt.replace("{Word}", parts[0])
            input_text = input_text.replace("{Sentence1}", parts[-2])
            input_text = input_text.replace("{Sentence2}", parts[-1])
            input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            labels = tokenizer(get_binary_label(self.wic_gold[idx]), padding='max_length',truncation=True,max_length=4,return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'wic'