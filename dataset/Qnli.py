from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label ==1:
        return "No"
    elif label ==0:
        return "Yes"
    else:
        raise ValueError("Invalid label")

def set_flan_prompt(idx = 0):
    prompt = [
    """{Premise}\nBased on the paragraph above can we answer that "{hypothesis}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-Yes. \n-No. """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """qnli premise: {Premise}\n "qnli hypothesis: {hypothesis}""",
]
    return prompt[idx]

class Qnli(Dataset):
    def __init__(self, train=True, stop_flan=False,prompt_idx=0):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading QNLI")
        qnli = load_dataset("nyu-mll/glue", "qnli")

        if train:
            self.qnli = qnli["train"].select(range(10000))
            
        else:
            self.qnli = qnli["validation"].select(range(1000))
        

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.qnli:
            input_text = prompt.replace("{Premise}", sample['sentence'])
            input_text = input_text.replace("{hypothesis}", sample['question'])
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
        return sentence, target , 'qnli'