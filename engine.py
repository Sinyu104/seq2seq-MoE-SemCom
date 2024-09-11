import torch
from utils import task_metrics_mapping
from transformers import AutoTokenizer
from torch.cuda.amp import autocast
from dataset.MMLU import subcategories
import numpy as np





def evaluate(args, model, testloader, device, print_freq=10):
    metrics = task_metrics_mapping(args)
    scores = {task: 0 for task in args.test_task}
    cr = {task: 0 for task in args.test_task}
    loss = {task: 0 for task in args.test_task}
    cr_batch = {task: 0 for task in args.test_task}
    total = {task: 0 for task in args.test_task}
    final = {'loss': {task: 0 for task in args.test_task}, 'score': {task: 0 for task in args.test_task}, 'compression rate':{task: 0 for task in args.test_task}}
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model.eval()
    for task in args.test_task:
        if task.lower() == 'mmlu':
            subjects = list(subcategories.keys())
            score_per_subject = {subject: 0 for subject in subjects}
            batch_per_subject = {subject: 0 for subject in subjects}
            for subject in subjects:
                for batch_idx, data in enumerate(testloader['mmlu'][subject]):
                    texts, masks = data[0]['input_ids'].squeeze(1).to(device, non_blocking=True), data[0]['attention_mask'].squeeze(1).to(device, non_blocking=True)
                    
                    targets = data[1].squeeze(1).to(device, non_blocking=True)
                    batch_size = targets.shape[0]
                    labels = []
                    for token_ids in targets:
                        labels.append(tokenizer.decode(token_ids, skip_special_tokens=True))
                    # compression_rate = model.get_compression_rate(input_ids=texts, attention_mask=masks)
                    # cr['qa'] += compression_rate.item()
                    # cr_batch['qa'] += 1
                    decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.expand(batch_size, 1).to(device)
                    decoder_input_ids = model._shift_right(decoder_input_ids)
                    logits = model(
                        input_ids=texts, decoder_input_ids=decoder_input_ids
                    ).logits
                    probs = (
                        torch.nn.functional.softmax(
                            torch.index_select(
                                logits, 2,
                                torch.tensor(
                                    [
                                        tokenizer("A").input_ids[0],
                                        tokenizer("B").input_ids[0],
                                        tokenizer("C").input_ids[0],
                                        tokenizer("D").input_ids[0],
                                    ]
                                ).to(device)
                                
                            ),
                            dim=2,
                        )
                    )
                    max_indices = torch.argmax(probs, dim=2).squeeze().cpu().numpy().tolist() 
                    max_indices = max_indices if isinstance(max_indices,list) else [max_indices]
                    predicted_labels = [["A", "B", "C", "D"][index] for index in max_indices]
                    

                    result = metrics['mmlu'].compute(predictions=predicted_labels, references=labels)
                    scores['mmlu'] += result["exact_match"]
                    total['mmlu'] += 1
                    score_per_subject[subject] += result["exact_match"]
                    batch_per_subject[subject] += 1
                    if batch_idx % print_freq == 0:
                        print('[MMLU][%s] Test %d/%d: [score: %f] ' %(subject, batch_idx*batch_size, len(testloader['mmlu'][subject].dataset), score_per_subject[subject]/batch_per_subject[subject]))# , cr['qa']/cr_batch['qa']
            print('[MMLU][Average] Test %d subjects: [score: %f] ' %( len(subjects), scores['mmlu']/total['mmlu']))
        else:
            with torch.no_grad():
                for batch_idx, data in enumerate(testloader[task]):
                    texts, masks = data[0]['input_ids'].squeeze(1).to(device, non_blocking=True), data[0]['attention_mask'].squeeze(1).to(device, non_blocking=True)
                    targets = data[1].squeeze(1).to(device, non_blocking=True)
                    batch_size = targets.shape[0]
                    compression_rate = model.get_compression_rate(input_ids=texts, attention_mask=masks)
                    cr['sen'] += compression_rate.item()
                    cr_batch['sen'] += 1
                    outputs = model.generate(input_ids=texts, attention_mask=masks)
                    
                    for ii in range(batch_size):
                        predicted = tokenizer.decode(outputs[ii], skip_special_tokens=True)
                        labels = tokenizer.decode(targets[ii], skip_special_tokens=True)
                        result = metrics['sen'].compute(predictions=[predicted], references=[labels])
                        scores['sen'] += result["exact_match"]
                        total['sen'] += 1
                    if batch_idx % print_freq == 0:
                        print('[SEN] Test %d/%d: [score: %f]' %(batch_idx*batch_size, len(testloader['sen'].dataset), scores['sen']/total['sen']))# , cr['sen']/cr_batch['sen']
        elif task.lower() == 'trans':
            with torch.no_grad():
                for batch_idx, data in enumerate(testloader['trans']):
                    texts, masks = data[0]['input_ids'].squeeze().to(device, non_blocking=True), data[0]['attention_mask'].squeeze().to(device, non_blocking=True)
                    targets = data[1].squeeze().to(device, non_blocking=True)
                    batch_size = targets.shape[0]
                    # compression_rate = model.get_compression_rate(input_ids=texts, attention_mask=masks)
                    # cr['trans'] += compression_rate.item()
                    # cr_batch['trans'] += 1
                    outputs = model.generate(input_ids=texts, attention_mask=masks, max_length=64)
                    for ii in range(batch_size):
                        predicted = tokenizer.decode(outputs[ii], skip_special_tokens=True)
                        # print("predicted: ", predicted)
                        labels = tokenizer.decode(targets[ii], skip_special_tokens=True)
                        # print("labels: ", labels)
                        result = metrics['trans'].compute(predictions=[predicted], references=[[labels]])
                        # print("Result: ", result["score"])
                        # input("Predict")
                        scores['trans'] += result["score"]
                        total['trans'] += 1
                    if batch_idx % print_freq == 0:
                        print('[TRANS] Test %d/%d: [score: %f] ' %(batch_idx*batch_size, len(testloader['trans'].dataset), scores['trans']/total['trans']))# , cr['trans']/cr_batch['trans']
        elif task.lower() == 'qa':
            for batch_idx, data in enumerate(testloader['qa']):
                    texts, masks = data[0]['input_ids'].squeeze().to(device, non_blocking=True), data[0]['attention_mask'].squeeze().to(device, non_blocking=True)
                    targets = data[1].squeeze().to(device, non_blocking=True)
                    batch_size = targets.shape[0]
                    # compression_rate = model.get_compression_rate(input_ids=texts, attention_mask=masks)
                    # cr['qa'] += compression_rate.item()
                    # cr_batch['qa'] += 1
                    outputs = model.generate(input_ids=texts, attention_mask=masks, max_length=32)
                    for ii in range(batch_size):
                        predicted = tokenizer.decode(outputs[ii], skip_special_tokens=True)
                        # print("predicted: ", predicted)
                        labels = tokenizer.decode(targets[ii], skip_special_tokens=True)
                        # print("labels: ", labels)
                        result = metrics['qa'].compute(predictions=[predicted], references=[[labels]])
                        # print("Result: ", result['rouge1'])
                        # input("Predict")
                        scores['qa'] += result["rouge1"]
                        total['qa'] += 1
                    if batch_idx % print_freq == 0:
                        print('[QA] Test %d/%d: [score: %f] ' %(batch_idx*batch_size, len(testloader['qa'].dataset), scores['qa']/total['qa']))# , cr['qa']/cr_batch['qa']
        else:
            raise NotImplementedError
    for task in args.test_task:
        final['loss'][task] = loss[task]/cr_batch[task]
        final['score'][task] = scores[task]/total[task]
        final['compression rate'][task] = cr[task]/cr_batch[task]
    return final


def train(args, model, dataloader, optimizer, loss_scaler, device, mode, print_freq=20, accumulation_steps=2):
    total_loss = 0.0
    cr = {task: 0 for task in args.train_task}
    task_batch = {task: 0 for task in args.train_task}
    model.train(True)
    optimizer.zero_grad()
    running_loss = torch.tensor(0.0).to(device)

    for i, data_batch in enumerate(dataloader):
        texts, masks = data_batch[0]['input_ids'].squeeze(1).to(device), data_batch[0]['attention_mask'].squeeze(1).to(device)
        targets = data_batch[1].squeeze(1).to(device)
        task = data_batch[2][0]

        batch_size = targets.shape[0]

        with autocast(enabled=False):
            outputs = model(input_ids=texts, attention_mask=masks, labels=targets, mode=mode, task=task)
            loss = outputs.loss
        
        compression_rate = outputs.compression_rate
        cr[task]+=compression_rate.item()
        task_batch[task]+=1
        # print("CR task: ", task, cr[task], "task batch: ", task_batch[task])
        
        loss_scaler(i, loss, optimizer)
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if i % print_freq == 0:
            print('[Batch: %d/%d] [loss: %f] [compression rate: %f]' %(i+1, len(dataloader), total_loss/task_batch[task], cr[task]/task_batch[task]))
    
    avg_cr = {task: cr[task]/task_batch[task] for task in args.train_task}
        
    avg_train_loss = {task: task_loss[task]/task_batch[task] for task in args.train_task}
    return {
        'loss':avg_train_loss, 
        'compression_rate': avg_cr
    }
