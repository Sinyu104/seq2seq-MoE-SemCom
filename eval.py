from model import *
from transformers import AutoTokenizer
from transformers import AutoConfig
from loguru import logger
import pytreebank
import torch
import datetime
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric
import evaluate


config = AutoConfig.from_pretrained("google/flan-t5-small")

# logger.info("Loading SST")
# sst = pytreebank.load_sst()
# def get_binary_label(label):
#     """Convert fine-grained label to binary label."""
#     if label < 2:
#         return "negative"
#     if label > 2:
#         return "positive"
#     raise ValueError("Invalid label")
def set_prompt(idx = 0):
    prompt = [
    """Answer based on context:\n\n{context}\n\n{question}""",
    "translate English to French: ", 
    """Review:\n{sentence}\nIs this sentence negative or positive?\n{options_}"""
]
    return prompt[idx]

# def set_options():
#     options = """OPTIONS:\n-positive \n-negative """
#     return options

# class SST_CR(Dataset):
#     def __init__(self, train=True, idx = 0):
#         logger.info("Loading the tokenizer")
#         tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
#         logger.info("Loading SST")
        
#         if train:
#             self.sst = sst["train"]
            
#         else:
#             self.sst = sst["test"]
#         prompt = set_prompt(idx)
#         options = set_options()
#         self.data = []
#         for tree in self.sst:
#             if tree.label != 2:
#                 input_text = prompt.replace("{sentence}", tree.to_lines()[0])
#                 input_text = input_text.replace("{options_}", options)
#                 inputs = tokenizer(input_text, padding='max_length', return_tensors="pt")
#                 self.data.append((inputs, tokenizer(get_binary_label(tree.label), return_tensors="pt").input_ids))
                
                
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         sentence, target = self.data[index]
#         return sentence, target




class IWSLT_book(Dataset):
    def __init__(self, train=True, idx = 0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading IWSLT Books")
        books = load_dataset("iwslt2017", "iwslt2017-en-fr")
        if train:
            self.opus = books["train"]['translation'][:20000]
            
        else:
            self.opus = books["test"]['translation'][:5000]
        prompt = set_prompt(idx)
        self.data = []
        for example in self.opus:
            if len(tokenizer(example['en']).input_ids) > 20:
                continue
            input_text = prompt + example['en']
            inputs = tokenizer(input_text, padding='max_length', truncation=True, return_tensors="pt")
            self.data.append((inputs, tokenizer(example['fr'], padding='max_length', max_length=64, return_tensors="pt").input_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        return sentence, target

# class SQuAD(Dataset):
#     def __init__(self, train=True, idx = 0):
#         logger.info("Loading the tokenizer")
#         tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
#         logger.info("Loading SQuAD")
#         squad = load_dataset("squad", split="train[:10000]").train_test_split(test_size=0.2)
#         if train:
#             self.squad = squad["train"]
            
#         else:
#             self.squad = squad["test"]
        
#         prompt = set_prompt(idx)
#         self.data = []
#         for example in self.squad:
#             input_text = prompt.replace("{context}", example["context"])
#             input_text = input_text.replace("{question}", example["question"])
#             if len(tokenizer(input_text).input_ids)>256 or len(tokenizer(example["answers"]["text"][0]).input_ids)>32:
#                 continue
#             inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=256,return_tensors="pt")
#             self.data.append((inputs, tokenizer(example["answers"]["text"][0], padding='max_length', max_length=32, return_tensors="pt").input_ids))
            

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         sentence, target = self.data[index]
#         return sentence, target

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

testset = SQuAD(False)

testloader = torch.utils.data.DataLoader(dataset=testset, num_workers=4, pin_memory=True,
                                                batch_size=40, shuffle=False,
                                                drop_last = True)

print("Creating model: Flan-T5")
model = T5SC_model.from_pretrained(pretrained_model_name_or_path="google/flan-t5-small", config=config)
model.initial_weights()


# Stop gradient for pre-trained model
for name, param in model.named_parameters():
    if not 'FSM' in name:
        param.requires_grad_(False)

# model.load_state_dict(torch.load('model_2024-04-18_00-18-04.pth'))
device = torch.device('cuda')
model.to(device)

metric = evaluate.load('rouge')
score = 0.0
total = 0


with torch.no_grad():
    for i, data in enumerate(testloader):
        # print("data: ", data[0]['input_ids'].squeeze())
        # print("target: ", data[1].squeeze())
        texts, masks = data[0]['input_ids'].squeeze().to(device, non_blocking=True), data[0]['attention_mask'].squeeze().to(device, non_blocking=True)
        targets = data[1].squeeze().to(device, non_blocking=True)
        batch_size = targets.shape[0]
        
        outputs = model.generate(input_ids=texts, attention_mask=masks)
        for ii in range(batch_size):
            predicted = tokenizer.decode(outputs[ii], skip_special_tokens=True)
            # print("predicted: ", predicted)
            labels = tokenizer.decode(targets[ii], skip_special_tokens=True)
            # print("labels: ", labels)
            result = metric.compute(predictions=[predicted], references=[[labels]])
            # print("Result: ", result['rouge1'])
            # input("Predict")
            score += result['rouge1']
            total += 1
        

print("Score: ", score/total)