from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from loguru import logger

def set_flan_prompt(idx = 0):
    prompt = [
    """Answer based on context:\n\n{context}\n\n{question}"""
]
    return prompt[idx]

def set_prompt(idx = 0):
    prompt = [
    """squad context: {context}\nsquad question: {question}"""
]
    return prompt[idx]



class SQuAD(Dataset):
    def __init__(self, train=True, stop_flan=False,idx = 0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading SQuAD dataset")
        squad = load_dataset("squad", split="train[:10000]").train_test_split(test_size=0.2)
        if train:
            self.squad = squad["train"]
            
        else:
            self.squad = squad["test"]
        
        if not stop_flan:
            prompt = set_flan_prompt(idx)
        else:
            prompt = set_prompt(idx)
        self.data = []
        for example in self.squad:
            input_text = prompt.replace("{context}", example["context"])
            input_text = input_text.replace("{question}", example["question"])
            if len(tokenizer(input_text).input_ids)>256 or len(tokenizer(example["answers"]["text"][0]).input_ids)>32:
                continue
            inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
            labels = tokenizer(example["answers"]["text"][0], padding='max_length', max_length=32, return_tensors="pt").input_ids
            if train:
                labels[labels == tokenizer.pad_token_id] = -100
            self.data.append((inputs, labels))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target , 'qa'