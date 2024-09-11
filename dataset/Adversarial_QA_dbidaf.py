from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch


def set_prompt(idx = 0):
    prompt = [
     """Context:{context}\nExtract the answer to the question from the following context.\nQuestion:{Question}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{options_1}\n-{options_2}\n-{options_3}\n-{options_4}""",
    ]
    return options[idx]


class Adversarial_QA_dbidaf(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading adversarial_qa_dbidaf dataset")
        adversarial_qa = load_dataset("UCLNLP/adversarial_qa", "dbidaf")
        
        if train:
            self.adversarial_qa = adversarial_qa["train"]
            
        else:
            self.adversarial_qa = adversarial_qa["validation"].select(range(1000))
        
        prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.adversarial_qa:
            input_text = prompt.replace("{context}", sample['context'])
            input_text = input_text.replace("{Question}", sample['question'])
            
            inputs = tokenizer(input_text, padding='max_length', max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            labels = tokenizer(sample['answers']['text'], padding='max_length',truncation=True,max_length=64,return_tensors="pt")
            if torch.sum(labels['attention_mask']) >=64:
                continue
            labels = labels.input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'adversarial_qa_dbidaf'
