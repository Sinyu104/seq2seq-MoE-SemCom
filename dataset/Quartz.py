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
    """{Information}\nBased on the paragraph above can we answer the question that "{question}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n{options_1}\n-{options_2} """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """quartz information: {Information}\n quartz question: {question}""",
]
    return prompt[idx]

class Quartz(Dataset):
    def __init__(self, train=True, stop_flan=False,prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading Quartz dataset")
        quartz = load_dataset("allenai/quartz")

        if train:
            self.quartz = quartz["train"]
            
        else:
            self.quartz = quartz["test"]
        

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.quartz:
            input_text = prompt.replace("{Information}", sample['para'])
            input_text = input_text.replace("{question}", sample['question'])
            if not stop_flan:
                input_text = input_text.replace("{options_}", options)
                input_text = input_text.replace("{options_1}", sample['choices']['text'][0])
                input_text = input_text.replace("{options_2}", sample['choices']['text'][1])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            if sample['answerKey']== 'A':
                labels = tokenizer(sample['choices']['text'][0], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            elif sample['answerKey']=='B':
                labels = tokenizer(sample['choices']['text'][1], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'quartz'