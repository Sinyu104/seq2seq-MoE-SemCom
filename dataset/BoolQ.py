from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label == True:
        return "Yes"
    elif label == False:
        return "No"
    else:
        raise ValueError("Invalid label")

def set_prompt(idx = 0):
    prompt = [
    """{Passage}\nBased on the paragraph above can we answer the question that "{question}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-Yes.\n-No. """,
    ]
    return options[idx]


class BoolQ(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading boolq dataset")
        boolq = load_dataset("google/boolq")

        if train:
            self.boolq = boolq["train"]
            
        else:
            self.boolq = boolq["validation"].select(range(1000))
        

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.boolq:
            input_text = prompt.replace("{Passage}", sample['passage'])
            input_text = input_text.replace("{question}", sample['question'])
            input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            labels = tokenizer(get_binary_label(sample['answer']), padding='max_length',truncation=True,max_length=4,return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'boolq'