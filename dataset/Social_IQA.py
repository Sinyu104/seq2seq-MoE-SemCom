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
    """{context}\nBased on the paragraph above can we answer the question that "{question}".?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n{options_1}\n-{options_2}n-{options_3} """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """social_iqa context: {context}\n social_iqa question:{question}""",
]
    return prompt[idx]

class Social_IQA(Dataset):
    def __init__(self, train=True, stop_flan=False,prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading Social IQA dataset")
        social_i_qa = load_dataset("allenai/social_i_qa")

        if train:
            self.social_i_qa = social_i_qa["train"].select(range(10000))
            
        else:
            self.social_i_qa = social_i_qa["validation"].select(range(1000))
        

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.social_i_qa:
            input_text = prompt.replace("{context}", sample['context'])
            input_text = input_text.replace("{question}", sample['question'])
            if not stop_flan:
                input_text = input_text.replace("{options_}", options)
                input_text = input_text.replace("{options_1}", sample['answerA'])
                input_text = input_text.replace("{options_2}", sample['answerB'])
                input_text = input_text.replace("{options_3}", sample['answerC'])
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            if torch.sum(inputs['attention_mask']) >=256:
                continue
            if sample['label']== '1':
                labels = tokenizer(sample['answerA'], padding='max_length',truncation=True,max_length=16,return_tensors="pt")
            elif sample['label']=='2':
                labels = tokenizer(sample['answerB'], padding='max_length',truncation=True,max_length=16,return_tensors="pt")
            elif sample['label']=='3':
                labels = tokenizer(sample['answerC'], padding='max_length',truncation=True,max_length=16,return_tensors="pt")
            if torch.sum(labels['attention_mask']) >=16:
                continue
            labels = labels.input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'social_i_qa'
