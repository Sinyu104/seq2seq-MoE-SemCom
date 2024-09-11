from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch


def set_prompt(idx = 0):
    prompt = [
     """Read the below conversation.\n{Dialogue}\n{Question}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{options_1}\n-{options_2}\n-{options_3}""",
    ]
    return options[idx]


class Dream(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading Dream dataset")
        dream = load_dataset("dataset-org/dream")
        
        if train:
            self.dream = dream["train"]
            
        else:
            self.dream = dream["test"].select(range(1000))
        
        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.dream:
            dialogue = merged_dialogue = ' '.join(sample['dialogue'])
            input_text = prompt.replace("{Dialogue}", dialogue)
            input_text = input_text.replace("{Question}", sample['question'])
            input_text = input_text.replace("{options_}", options)
            input_text = input_text.replace("{options_1}", sample['choice'][0])
            input_text = input_text.replace("{options_2}", sample['choice'][1])
            input_text = input_text.replace("{options_3}", sample['choice'][2])
            
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(sample['answer'], padding='max_length',truncation=True,max_length=36,return_tensors="pt")
            if torch.sum(labels['attention_mask']) >=36:
                continue
            labels = labels.input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'dream'
