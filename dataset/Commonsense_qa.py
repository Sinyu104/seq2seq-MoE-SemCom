from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch


def set_prompt(idx = 0):
    prompt = [
     """{Question}\nChoose the most suitable option to answer the above question.\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{options_1}\n-{options_2}\n-{options_3}\n-{options_4}\n-{options_5}""",
    ]
    return options[idx]


class Commonsense_QA(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading commonsense_qa dataset")
        commonsense_qa = load_dataset("tau/commonsense_qa")
                
        if train:
            self.commonsense_qa = commonsense_qa["train"]
            
        else:
            self.commonsense_qa = commonsense_qa["validation"].select(range(1000))
        
        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.commonsense_qa:
            input_text = prompt.replace("{Question}", sample['question'])
            input_text = input_text.replace("{options_}", options)
            input_text = input_text.replace("{options_1}", sample['choices']['text'][0])
            input_text = input_text.replace("{options_2}", sample['choices']['text'][1])
            input_text = input_text.replace("{options_3}", sample['choices']['text'][2])
            input_text = input_text.replace("{options_4}", sample['choices']['text'][3])
            input_text = input_text.replace("{options_5}", sample['choices']['text'][4])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            if sample['answerKey']== 'A':
                labels = tokenizer(sample['choices']['text'][0], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            elif sample['answerKey']=='B':
                labels = tokenizer(sample['choices']['text'][1], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            elif sample['answerKey']=='C':
                labels = tokenizer(sample['choices']['text'][2], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            elif sample['answerKey']=='D':
                labels = tokenizer(sample['choices']['text'][3], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            elif sample['answerKey']=='E':
                labels = tokenizer(sample['choices']['text'][4], padding='max_length',truncation=True,max_length=16,return_tensors="pt").input_ids
            else:
                continue
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'commonsense_qa'
