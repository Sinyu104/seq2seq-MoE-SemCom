from datasets import load_dataset, concatenate_datasets
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

def set_prompt(idx = 0):
    prompt = [
    """{Premise}\nBased on the paragraph above can we conclude that "{hypothesis}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-Yes. \n-It's impossible to say.\n-No. """,
    ]
    return options[idx]


class Anli(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading facebook/anli dataset")
        anli = load_dataset("facebook/anli")

        if train:
            self.anli = concatenate_datasets([anli["train_r1"].select(range(5000)), anli["train_r2"].select(range(5000)), anli["train_r3"].select(range(5000))])
            
        else:
            self.anli = concatenate_datasets([anli["test_r1"].select(range(1000)), anli["test_r2"].select(range(1000)), anli["test_r3"].select(range(1200))])
        

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.anli:
            input_text = prompt.replace("{Premise}", sample['premise'])
            input_text = input_text.replace("{hypothesis}", sample['hypothesis'])
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
        return sentence, target , 'anli'