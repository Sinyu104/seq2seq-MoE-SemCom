from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label, idx):
    """Convert fine-grained label to binary label."""
    if label == 1:
        return "yes"
    else:
        return "no"
    raise ValueError("Invalid label")

def set_prompt(idx = 0):
    prompt = [
    """Determine if the following two sentences paraphrase each other or not.\nSent 1: {{sentence1}}\nSent 2: {{sentence2}}\n{options_}""",
    """Sentence 1: {{sentence1}}\nSentence 2: {{sentence2}}\nQuestion: Do Sentence 1 and Sentence 2 express the same meaning? Yes or No?""",
    """{{sentence1}}\nIs that a paraphrase of the following sentence?\n{{sentence2}}?\n{options_}""",
    """Sentence 1: {{sentence1}}\nSentence 2: {{sentence2}}\nQuestion: Can we rewrite Sentence 1 to Sentence 2?\n{options_}""",
    """{{sentence1}}\nIs that a paraphrase of the following sentence?\n{{sentence2}}?\nYes or No.\n{options_}""",
    """Sentence 1: {{sentence1}}\nSentence 2: {{sentence2}}\nQuestion: Does Sentence 1 paraphrase Sentence 2? Yes or No?\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-no \n-yes """,
        """""",
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-no \n-yes """,
        """""",
        """""",
    ]
    return options[idx]


class labeled_final(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading paws/labeled_final dataset")

        labeled_final = load_dataset("google-research-datasets/paws", "labeled_final")

        if train:
            self.labeled_final = labeled_final["train"].select(range(1000))
            
        else:
            self.labeled_final = labeled_final["test"].select(range(2000))

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.labeled_final:
            input_text = prompt.replace("{sentence1}", sample['sentence1'])
            input_text = input_text.replace("{sentence2}", sample['sentence2'])
            input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            self.data.append((inputs, tokenizer(get_binary_label(sample['label'], prompt_idx), return_tensors="pt").input_ids))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'labeled_final'
        
    


