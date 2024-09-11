from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch


def set_prompt(idx = 0):
    prompt = [
     """{document}\nGenerate a title for this article:""",
]
    return prompt[idx]


class Gigaword(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading gigaword dataset")
        gigaword = load_dataset("Harvard/gigaword")
        

        if train:
            self.gigaword = gigaword["train"].select(range(10000))
            
        else:
            self.gigaword = gigaword["test"].select(range(1000))
        
        prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.gigaword:
            
            input_text = prompt.replace("{document}", sample['document'])
            inputs = tokenizer(input_text, padding='max_length', max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(sample['summary'], padding='max_length',truncation=True,max_length=64,return_tensors="pt")
            if torch.sum(labels['attention_mask']) >=64:
                continue
            labels = labels.input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'gigaword'
    
