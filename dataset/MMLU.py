from datasets import Dataset as DS
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger
from collections import defaultdict
import pandas as pd
import os

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, include_answer=True):
    prompt = df[0]
    options = 4
    prompt += "\nOPTIONS:"
    for j in range(options):
        prompt += "\n{}".format(df[j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}".format(df[-1])
    return prompt


def gen_prompt(subject):
    prompt = "Please answer the multiple choice questions about {}.\n".format(
        format_subject(subject)
    )
    return prompt


class MMLU(Dataset):
    def __init__(self, train=True):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading MMLU dataset")
        subjects = list(subcategories.keys())
        
        if train:
            self.mmlu = []
            self.data = []
            for subject in subjects:
                file_path = os.path.join("dataset\\MMLU_data", "dev", subject + "_dev.csv")
                df = pd.read_csv(file_path, header=None)
                self.mmlu.append(df)
                prompt = gen_prompt(subject)
                for index, example in df.iterrows():
                    input_text = prompt+format_example(example, include_answer=False)
                    inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
                    labels = tokenizer(str(example[5]), padding='max_length', max_length=8, return_tensors="pt").input_ids
                    labels[labels == tokenizer.pad_token_id] = -100
                    self.data.append((inputs, labels))
            self.mmlu = pd.concat(self.mmlu, ignore_index=True)
            self.mmlu =DS.from_pandas(self.mmlu)

              
        else:
            self.mmlu = []
            self.data = defaultdict(list)
            for subject in subjects:
                file_path = os.path.join("dataset\\MMLU_data", "test", subject + "_test.csv")
                df = pd.read_csv(file_path, header=None)
                prompt = gen_prompt(subject)
                for index, example in df.iterrows():
                    input_text = prompt+format_example(example, include_answer=False)
                    inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
                    labels = tokenizer(str(example[5]), padding='max_length', max_length=8, return_tensors="pt").input_ids
                    labels[labels == tokenizer.pad_token_id] = -100
                    self.data[subject].append((inputs, labels))
                self.mmlu.append(df)
            self.mmlu = pd.concat(self.mmlu, ignore_index=True)
            self.mmlu.sample(n=1000)
            self.mmlu =DS.from_pandas(self.mmlu)
        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if type(index)==int:
            sentence, target = self.data[index]
        else:
            subject, idx = index
            sentence, target = self.data[subject][idx]
        return sentence, target , 'mmlu'
    


