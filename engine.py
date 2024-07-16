import torch
from utils import task_metrics_mapping
from transformers import AutoTokenizer
from torch.cuda.amp import autocast





def evaluate(args, model, testloader, device, print_freq=10):
    metrics = task_metrics_mapping(args)
    scores = {task: 0 for task in args.test_task}
    cr = {task: 0 for task in args.test_task}
    cr_batch = {task: 0 for task in args.test_task}
    total = {task: 0 for task in args.test_task}
    final = {'score': {task: 0 for task in args.test_task}, 'compression rate':{task: 0 for task in args.test_task}}
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model.eval()
    for task in args.test_task:
        if task.lower() == 'sen':
            with torch.no_grad():
                for batch_idx, data in enumerate(testloader['sen']):
                    texts, masks = data[0]['input_ids'].squeeze().to(device, non_blocking=True), data[0]['attention_mask'].squeeze().to(device, non_blocking=True)
                    targets = data[1].squeeze().to(device, non_blocking=True)
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
                        print('[SEN] Test %d/%d: [score: %f] [compress rate: %f]' %(batch_idx*batch_size, len(testloader['sen'].dataset), scores['sen']/total['sen'], cr['sen']/cr_batch['sen']))
        elif task.lower() == 'trans':
            with torch.no_grad():
                for batch_idx, data in enumerate(testloader['trans']):
                    texts, masks = data[0]['input_ids'].squeeze().to(device, non_blocking=True), data[0]['attention_mask'].squeeze().to(device, non_blocking=True)
                    targets = data[1].squeeze().to(device, non_blocking=True)
                    batch_size = targets.shape[0]
                    compression_rate = model.get_compression_rate(input_ids=texts, attention_mask=masks)
                    cr['trans'] += compression_rate.item()
                    cr_batch['trans'] += 1
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
                        print('[TRANS] Test %d/%d: [score: %f] [compress rate: %f]' %(batch_idx*batch_size, len(testloader['trans'].dataset), scores['trans']/total['trans'], cr['trans']/cr_batch['trans']))
        elif task.lower() == 'qa':
            for batch_idx, data in enumerate(testloader['qa']):
                    texts, masks = data[0]['input_ids'].squeeze().to(device, non_blocking=True), data[0]['attention_mask'].squeeze().to(device, non_blocking=True)
                    targets = data[1].squeeze().to(device, non_blocking=True)
                    batch_size = targets.shape[0]
                    compression_rate = model.get_compression_rate(input_ids=texts, attention_mask=masks)
                    cr['qa'] += compression_rate.item()
                    cr_batch['qa'] += 1
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
                        print('[QA] Test %d/%d: [score: %f] [compress rate: %f]' %(batch_idx*batch_size, len(testloader['qa'].dataset), scores['qa']/total['qa'], cr['qa']/cr_batch['qa']))
        else:
            raise NotImplementedError
    for task in args.test_task:
        final['score'][task] = scores[task]/total[task]
        final['compression rate'][task] = cr[task]/cr_batch[task]
    return final


def train(args, model, dataloader, optimizer, loss_scaler, device, mode, print_freq=20, accumulation_steps=1):
    task_loss = {task: 0 for task in args.train_task}
    cr = {task: 0 for task in args.train_task}
    task_batch = {task: 0 for task in args.train_task}
    model.train()
    optimizer.zero_grad()
    running_loss = torch.tensor(0.0).to(device)

    for i, data_batch in enumerate(dataloader):
        
        texts, masks = data_batch[0]['input_ids'].squeeze().to(device), data_batch[0]['attention_mask'].squeeze().to(device)
        targets = data_batch[1].squeeze().to(device)
        task = data_batch[2][0]

        batch_size = targets.shape[0]

        
        with autocast(enabled=False):
            outputs = model(input_ids=texts, attention_mask=masks, labels=targets, mode=mode, task=task)
        
            loss = outputs.loss
        
        compression_rate = outputs.compression_rate
        cr[task]+=compression_rate.item()
        task_batch[task]+=1
        # print("CR task: ", task, cr[task], "task batch: ", task_batch[task])
        loss += running_loss.item()
        # loss_scaler(i, loss, optimizer)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        task_loss[task] += loss.item()

        if (i + 1) % accumulation_steps != 0:
            running_loss += loss
        else:
            running_loss = torch.tensor(0.0).to(device)
        if (i + 1) % print_freq == 0:
            print('[Batch: %d/%d] [loss: %f] [compression rate: %f]' %(i+1, len(dataloader), task_loss[task]/task_batch[task], cr[task]/task_batch[task]))

    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    avg_cr = {task: cr[task]/task_batch[task] for task in args.train_task}
        
    avg_train_loss = {task: task_loss[task]/task_batch[task] for task in args.train_task}
    return {
        'loss':avg_train_loss, 
        'compression_rate': avg_cr
    }
