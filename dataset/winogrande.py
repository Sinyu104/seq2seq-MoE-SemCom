from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch



def set_prompt(idx = 0):
    prompt = [
    """{Sentence}\nBased on the sentence provided, determine which option should fill in the blank.\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-{options_1}\n-{options_2}""",
    ]
    return options[idx]


class Winogrande(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading hellaswag dataset")
        winog = load_dataset("allenai/winogrande", 'winogrande_l')
        
        if train:
            self.winog = winog["train"]
            
        else:
            self.winog = winog["validation"]
        
        
        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.winog:
            input_text = prompt.replace("{Sentence}", sample['sentence'])
            input_text = input_text.replace("{options_}", options)
            input_text = input_text.replace("{options_1}", sample['option1'])
            input_text = input_text.replace("{options_2}", sample['option2'])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            if sample['answer']== '1':
                labels = tokenizer(sample['option1'], padding='max_length',truncation=True,max_length=8,return_tensors="pt").input_ids
            elif sample['answer']=='2':
                labels = tokenizer(sample['option2'], padding='max_length',truncation=True,max_length=8,return_tensors="pt").input_ids
            else:
                continue

            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'winog'