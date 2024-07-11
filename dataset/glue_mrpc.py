from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
import torch

def get_binary_label(label, idx):
    """Convert fine-grained label to binary label."""
    if idx == 2:
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

def set_prompt(idx = 0):
    prompt = [
    """I want to know whether the following two sentences mean the same thing.\n{{sentence1}}\n{{sentence2}}Do they?\n{options_}""",
    """Does the sentence\n{{sentence1}}\nparaphrase (that is, mean the same thing as) this sentence?\n{{sentence2}}\n{options_}""",
    """Are the following two sentences "{{"equivalent"}}" or "{{"not equivalent"}}"? {{sentence1}} {{sentence2}}\n{options_}""",
    """Can I replace the sentence{{sentence1}}with the sentence{{sentence2}}and have it mean the same thing?\n{options_}""",
    """Do the following two sentences mean the same thing?\n{{sentence1}}\n{{sentence2}}\n{options_}""",
]
    return prompt[idx]

def set_options(idx = 0):
    options = [
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-not equivalent \n-equivalent """,
        """OPTIONS:\n-no \n-yes """,
        """OPTIONS:\n-no \n-yes """,
    ]
    return options[idx]


class Glue_mrpc(Dataset):
    def __init__(self, train=True, prompt_idx=0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading glue/mrpc dataset")
        mrpc = load_dataset("nyu-mll/glue", "mrpc")

        if train:
            self.mrpc = mrpc["train"]
            
        else:
            self.mrpc = mrpc["test"]

        prompt = set_prompt(prompt_idx)
        options = set_options(prompt_idx)
        self.data = []
        for sample in self.mrpc:
            input_text = prompt.replace("{sentence1}", sample['sentence1'])
            input_text = input_text.replace("{sentence2}", sample['sentence2'])
            input_text = input_text.replace("{options_}", options)
            inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
            self.data.append((inputs, tokenizer(get_binary_label(sample['label'], prompt_idx), return_tensors="pt").input_ids))
            # print("data: ",(inputs, tokenizer(get_binary_label(tree.label), return_tensors="pt").input_ids) )
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'glue_mrpc'
        

m = Glue_mrpc()
    


