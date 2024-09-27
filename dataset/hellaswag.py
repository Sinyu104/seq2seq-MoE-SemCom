from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label, endings):
    """Convert fine-grained label to binary label."""
    interger_label = int(label)
    if interger_label >=0 and interger_label < 4:
        return endings[interger_label]
    else:
        raise ValueError("Invalid label")

def set_prompt(idx = 0):
    prompt = [
    """What happens next in this paragraph?\n{Passage}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{options_1}\n-{options_2}\n-{options_3}\n-{options_4}""",
    ]
    return options[idx]


class HellaSwag(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading hellaswag dataset")
        hella = load_dataset("Rowan/hellaswag")

        if train:
            self.hella = hella["train"].select(range(10000))
            
        else:
            self.hella = hella["validation"].select(range(1000))
        

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.hella:
            input_text = prompt.replace("{Passage}", sample['ctx'])
            input_text = input_text.replace("{options_}", options)
            input_text = input_text.replace("{options_1}", sample['endings'][0])
            input_text = input_text.replace("{options_2}", sample['endings'][1])
            input_text = input_text.replace("{options_3}", sample['endings'][2])
            input_text = input_text.replace("{options_4}", sample['endings'][3])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(get_binary_label(sample['label'], sample['endings']), padding='max_length',truncation=True,max_length=64,return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'hella'