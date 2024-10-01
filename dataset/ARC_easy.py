from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch



def set_flan_prompt(idx = 0):
    prompt = [
    """{Questions}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{Choise1}.\n-{Choise2}.\n-{Choise3}.\n-{Choise4}. """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """arc_easy questions:{Questions}\n""",
]
    return prompt[idx]


class ARC_easy(Dataset):
    def __init__(self, train=True, stop_flan=False,prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading arc dataset")
        arc = load_dataset("allenai/ai2_arc", "ARC-Easy")


        if train:
            self.arc = arc["train"]
            
        else:
            self.arc = arc["test"].select(range(1000))
            
        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.arc:
            input_text = prompt.replace("{Questions}", sample['question'])
            if not stop_flan:
                input_text = input_text.replace("{options_}", options)
                if not len(sample['choices']['text'])==4:
                    continue
                input_text = input_text.replace("{Choise1}", sample['choices']['text'][0])
                input_text = input_text.replace("{Choise2}", sample['choices']['text'][1])
                input_text = input_text.replace("{Choise3}", sample['choices']['text'][2])
                input_text = input_text.replace("{Choise4}", sample['choices']['text'][3])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if sample['answerKey']=='A' or sample['answerKey']=='1':
                labels = tokenizer(sample['choices']['text'][0], padding='max_length',truncation=True,max_length=32,return_tensors="pt").input_ids
            elif sample['answerKey']=='B'or sample['answerKey']=='2':
                labels = tokenizer(sample['choices']['text'][1], padding='max_length',truncation=True,max_length=32,return_tensors="pt").input_ids
            elif sample['answerKey']=='C'or sample['answerKey']=='3':
                labels = tokenizer(sample['choices']['text'][2], padding='max_length',truncation=True,max_length=32,return_tensors="pt").input_ids
            elif sample['answerKey']=='D'or sample['answerKey']=='4':
                labels = tokenizer(sample['choices']['text'][3], padding='max_length',truncation=True,max_length=32,return_tensors="pt").input_ids
            else:
                raise ValueError("Invalid label")
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'arc_easy'
