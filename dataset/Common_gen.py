from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch


def set_prompt(idx = 0):
    prompt = [
     """Generate a sentence with all the concepts:{Concepts}\n""",
]
    return prompt[idx]


class Common_gen(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading common_gen dataset")
        common_gen = load_dataset("GEM/common_gen")
                
        if train:
            self.common_gen = common_gen["train"].select(range(10000))
            
        else:
            self.common_gen = common_gen["validation"]
        
        prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.common_gen:
            concepts = ' '.join(sample['concepts'])
            input_text = prompt.replace("{Concepts}", concepts)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(sample['target'], padding='max_length', max_length=64, return_tensors="pt")
            if torch.sum(labels['attention_mask']) >=64:
                continue
            labels = labels["input_ids"]
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target, 'common_gen'