from transformers import AutoTokenizer
import pytreebank
from loguru import logger
from torch.utils.data import Dataset
import random



def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return "negative"
    if label > 2:
        return "positive"
    raise ValueError("Invalid label")


def set_flan_prompt(idx = 0):
    prompt = [
    """Review:\n{sentence}\nIs this review negative or positive?\n{options_}"""
]
    return prompt[idx]

def set_options():
    options = """OPTIONS:\n-positive \n-negative """
    return options

def set_prompt(idx = 0):
    prompt = [
    """sst2 sentence: {sentence}\n"""
]
    return prompt[idx]

class SST(Dataset):
    def __init__(self, train=True, stop_flan=False,idx = 0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading SST dataset")
        sst = pytreebank.load_sst()
        if train:
            self.sst = sst["train"]
            
        else:
            self.sst = sst["test"]
        
        if not stop_flan:
            prompt = set_flan_prompt(idx)
            options = set_options()
        else:
            prompt = set_prompt(idx)
        self.data = []
        for tree in self.sst:
            if tree.label != 2:
                input_text = prompt.replace("{sentence}", tree.to_lines()[0])
                if not stop_flan:
                    input_text = input_text.replace("{options_}", options)
                inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
                self.data.append((inputs, tokenizer(get_binary_label(tree.label), return_tensors="pt").input_ids))
                # print("data: ",(inputs, tokenizer(get_binary_label(tree.label), return_tensors="pt").input_ids) )
        random.shuffle(self.data)       
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'sen'