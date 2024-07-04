from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
from loguru import logger
import torch



def set_prompt(idx = 0):
    prompt = [
    "translate English to French: "
]
    return prompt[idx]

class IWSLT(Dataset):
    def __init__(self, train=True, idx = 0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading IWSLT Books dataset")
        books = load_dataset("iwslt2017", "iwslt2017-en-fr",trust_remote_code=True)
        if train:
            self.opus = books["train"]['translation'][:15000]
            
        else:
            self.opus = books["test"]['translation'][:5000]
        prompt = set_prompt(idx)
        self.data = []
        for example in self.opus:
            len_token = len(tokenizer(example['en']).input_ids)
            if len_token > 20 or len_token < 8:
                continue
            input_text = prompt + example['en']
            inputs = tokenizer(input_text, padding='max_length', max_length=256, return_tensors="pt")
            if len(tokenizer(example['fr']).input_ids) >64:
                continue
            labels = tokenizer(example['fr'], padding='max_length', max_length=64, return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'trans'