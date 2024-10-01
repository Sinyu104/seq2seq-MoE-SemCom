from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label, idx):
    """Convert fine-grained label to binary label."""
    if idx == 1:
        if label == 1:
            return "equivalent"
        else:
            return "not equivalent"
    else:
        if label == 1:
            return "yes"
        else:
            return "no"
    raise ValueError("Invalid label")

def set_flan_prompt(idx = 0):
    prompt = [
    """I'm an administrator on the website Quora. There are two posts, one that asks "{{question1}}" and another that asks "{{question2}}". I can merge questions if they are asking the same thing. Can I merge these two questions?\n{options_}""",
    """{{question1}} {{question2}} Pick one: These questions are "{{"duplicates"}}" or "{{"not duplicates"}}".\n{options_}""",
    """Are the questions "{{question1}}" and "{{question2}}" asking the same thing?\n{options_}""",
    """Can an answer to "{{question1}}" also be used to answer "{{question2}}"?\n{options_}""",
    """Question 1: {{question1}} Question 2: {{question2}} Do these two questions convey the same meaning? Yes or no?\n{options_}""",
    """I received the questions "{{question1}}" and "{{question2}}". Are they duplicates?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-not equivalent \n-equivalent """,
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-no \n-yes """,
    ]
    return options[idx]

def set_prompt(idx = 0):
    prompt = [
    """qqp question1: {{question1}} qqp question2: {{question2}}.""",
]
    return prompt[idx]


class Glue_qqp(Dataset):
    def __init__(self, train=True, stop_flan=False, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading glue/qqp dataset")
        qqp = load_dataset("nyu-mll/glue", "qqp")

        if train:
            self.qqp = qqp["train"].select(range(5000))
            
        else:
            self.qqp = qqp["test"].select(range(1000))

        if not stop_flan:
            prompt = set_flan_prompt(prompt_idx)
            options = set_options(prompt_idx)
        else:
            prompt = set_prompt(prompt_idx)
        self.data = []
        for sample in self.qqp:
            input_text = prompt.replace("{question1}", sample['question1'])
            input_text = input_text.replace("{question2}", sample['question2'])
            if not stop_flan:
                input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            self.data.append((inputs, tokenizer(get_binary_label(sample['label'], prompt_idx), return_tensors="pt").input_ids))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'glue_qqp'
        
    


