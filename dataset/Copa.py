from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch



def set_prompt(idx = 0):
    prompt = [
    """{premise}\nWhat is the "{question}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{Choise1}.\n-{Choise2}. """,
    ]
    return options[idx]


class Copa(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading copa dataset")
        copa = load_dataset("pkavumba/balanced-copa")


        if train:
            self.copa = copa["train"]
            
        else:
            self.copa = copa["test"]

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.copa:
            input_text = prompt.replace("{premise}", sample['premise'])
            input_text = input_text.replace("{question}", sample['question'])
            input_text = input_text.replace("{options_}", options)
            input_text = input_text.replace("{Choise1}", sample['choice1'])
            input_text = input_text.replace("{Choise2}", sample['choice2'])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if sample['label']==0:
                labels = tokenizer(sample['choice1'], padding='max_length',truncation=True,max_length=32,return_tensors="pt").input_ids
            elif sample['label']==1:
                labels = tokenizer(sample['choice2'], padding='max_length',truncation=True,max_length=32,return_tensors="pt").input_ids
            else:
                raise ValueError("Invalid label")
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'copa'
