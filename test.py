import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Dummy dataset for illustration
class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, summaries, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            summary, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens for loss calculation

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()

# Example data
texts = ["The quick brown fox jumps over the lazy dog."]
summaries = ["A fox jumps over a dog."]

# Create dataset and dataloader
dataset = TextDataset(tokenizer, texts, summaries)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer and GradScaler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)  # Lower learning rate
scaler = GradScaler()

# Training loop
num_epochs = 10

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        # Check for NaNs in inputs
        check_for_nans(input_ids, "input_ids")
        check_for_nans(attention_mask, "attention_mask")
        check_for_nans(labels, "labels")

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        # Check for NaNs in outputs
        check_for_nans(logits, "logits")

        if torch.isnan(loss):
            print("NaN found in loss")
            continue

        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training completed.")
