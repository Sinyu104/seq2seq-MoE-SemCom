from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label == 0:
        return "World politics"
    elif label == 1:
        return "Sports"
    elif label == 2:
        return "Business"
    elif label == 3:
        return "Science and technology"
    else:
        raise ValueError("Invalid label")

def set_prompt(idx = 0):
    prompt = [
    """What label best describes this news article?\n{Passage}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-World politics\n-Sports\n-Business\n-Science and technology""",
    ]
    return options[idx]


class Ag_news(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading boolq dataset")
        agnews = load_dataset("fancyzhx/ag_news")
        
        if train:
            self.agnews = agnews["train"].select(range(10000))
            
        else:
            self.agnews = agnews["test"].select(range(1000))
        

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.agnews:
            input_text = prompt.replace("{Passage}", sample['text'])
            input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(get_binary_label(sample['label']), padding='max_length',truncation=True,max_length=8,return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'agnews'