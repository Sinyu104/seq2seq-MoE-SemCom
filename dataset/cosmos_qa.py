from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label, idx):
    """Convert fine-grained label to binary label."""
    if idx == 1:
        if label == 1:
            return "equivalent"
        else:
            return "not equivalent"
    else:
        if label == 1:
            return "yes"
        else:
            return "no"
    raise ValueError("Invalid label")

def set_flan_prompt(idx = 0):
    prompt = [
    """ {Passage}\n Based on the passage, {Question}\n {options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{options_1}\n-{options_2}\n-{options_3}\n-{options_4}""",
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """cosmos_qa passage: {Passage}\n cosmos_qa question: {Question}""",
]
    return prompt[idx]


class Cosmos(Dataset):
    def __init__(self, train=True, stop_flan=False, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading allenai/cosmos_qa dataset")
        cosmos = load_dataset("allenai/cosmos_qa")

        if train:
            self.cosmos = cosmos["train"].select(range(10000))
            
        else:
            self.cosmos = cosmos["validation"].select(range(1500))

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.cosmos:
            
            input_text = prompt.replace("{Passage}", sample['context'])
            input_text = input_text.replace("{Question}", sample['question'])
            if not stop_flan:
                input_text = input_text.replace("{options_}", options)
                input_text = input_text.replace("{options_1}", sample['answer0'])
                input_text = input_text.replace("{options_2}", sample['answer1'])
                input_text = input_text.replace("{options_3}", sample['answer2'])
                input_text = input_text.replace("{options_4}", sample['answer3'])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            
            if sample['label']== 0:
                labels = tokenizer(sample['answer0'], padding='max_length',truncation=True,max_length=64,return_tensors="pt").input_ids
            elif sample['label']==1:
                labels = tokenizer(sample['answer1'], padding='max_length',truncation=True,max_length=64,return_tensors="pt").input_ids
            elif sample['label']==2:
                labels = tokenizer(sample['answer2'], padding='max_length',truncation=True,max_length=64,return_tensors="pt").input_ids
            elif sample['label']==3:
                labels = tokenizer(sample['answer3'], padding='max_length',truncation=True,max_length=64,return_tensors="pt").input_ids
            else:
                continue
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'cosmos'