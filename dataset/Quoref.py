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

def set_flan_prompt(idx = 0):
    prompt = [
    """{Context}\nBased on the paragraph about {Title} above can we answer the question that "{question}".?""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n{options_1}\n-{options_2}n-{options_3} """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """quoref context: {Context}\n quoref question:{question}""",
]
    return prompt[idx]

class Quoref(Dataset):
    def __init__(self, train=True, stop_flan=False,prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading Quoref dataset")
        quoref = load_dataset("allenai/quoref")

        if train:
            self.quoref = quoref["train"].select(range(10000))
            
        else:
            self.quoref = quoref["validation"].select(range(1000))
        

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.quoref:
            input_text = prompt.replace("{context}", sample['context'])
            input_text = input_text.replace("{Title}", sample['title'])
            input_text = input_text.replace("{question}", sample['question'])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(sample['answers']['text'][0], padding='max_length',truncation=True,max_length=32,return_tensors="pt")
            if torch.sum(labels['attention_mask']) >=32:
                continue
            labels = labels.input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'quoref'
