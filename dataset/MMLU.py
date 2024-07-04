from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from loguru import logger
from collections import defaultdict


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
    choices = df['choices']
    prompt = df['question']
    k = len(choices)
    prompt += "\nOPTIONS:"
    for j in range(k):
        prompt += "\n{}".format(choices[j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}".format(df['answer'])
    return prompt


def gen_prompt(train_df):
    subject = train_df[0]['subject']
    prompt = "Please answer the multiple choice questions about {}.\n".format(
        format_subject(subject)
    )
    return prompt

def def_value():
    return "Not Present"
class MMLU(Dataset):
    def __init__(self, train=True, idx = 0):
        logger.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        logger.info("Loading MMLU dataset")
        ds = load_dataset("cais/mmlu", "all")
        
        if train:
            self.mmlu = ds["dev"]
            
        else:
            self.mmlu = ds["test"]
        subjects = list(subcategories.keys())
        self.df = defaultdict(list)
        
        for example in self.mmlu:
            self.df[example['subject']].append(example)
        self.data = {subject: [] for subject in subjects}
        for subject in subjects:
            prompt = gen_prompt(self.df[subject])
            for example in self.df[subject]:
                input_text = prompt+format_example(example, include_answer=False)
                inputs = tokenizer(input_text, padding='max_length', truncation=True,max_length=256, return_tensors="pt")
                self.data[subject].append((inputs, tokenizer(str(example['answer']), return_tensors="pt").input_ids))
        
                
                
    def __len__(self):
        return sum(len(examples) for examples in self.data.values())

    def __getitem__(self, index):
        subject, idx = index
        sentence, target = self.data[subject][idx]
        return sentence, target , subject
    
c = MMLU()
print(len(c))
print(c[('clinical_knowledge',1)])


