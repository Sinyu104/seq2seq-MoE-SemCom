from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label == 2:
        return "No"
    elif label ==1:
        return "It's impossible to say"
    elif label ==0:
        return "Yes"
    else:
        raise ValueError("Invalid label")

def set_flan_prompt(idx = 0):
    prompt = [
    """{Premise}\nBased on the paragraph above can we conclude that "{hypothesis}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-Yes. \n-It's impossible to say.\n-No. """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """mlin premise: {Premise}\n mlin hypothesis: {hypothesis}?\n{options_}""",
]
    return prompt[idx]


class Mnli(Dataset):
    def __init__(self, train=True, stop_flan=False, prompt_idx=0):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading MNLI")
        mnli = load_dataset("nyu-mll/multi_nli")

        if train:
            self.mnli = mnli["train"].select(range(10000))
            
        else:
            self.mnli = mnli["validation_matched"].select(range(1000))
        

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.mnli:
            input_text = prompt.replace("{Premise}", sample['premise'])
            input_text = input_text.replace("{hypothesis}", sample['hypothesis'])
            if not stop_flan:
                input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            labels = tokenizer(get_binary_label(sample['label']), padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'mnli'