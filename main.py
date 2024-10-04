import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
from utils import *
from engine import evaluate, train
import numpy as np
from args import get_args
import torch
torch.autograd.set_detect_anomaly(True)
import sys
import time
import datetime
import random
from config import T5SC_config
# torch.set_printoptions(threshold=10_000)
from peft import LoraConfig, get_peft_model, TaskType
from dataset.MMLU import subcategories



def seed_inital(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def main(args):
    #### Get the Model 
    seed_inital(seed=args.seed)
    device=torch.device(args.device)
    config=T5SC_config()
    model = get_model(args, config=config)
    # Define LoRA Config
    # lora_config = LoraConfig(
    #     r=4,
    #     lora_alpha=32,
    #     target_modules=["q", "v","query","key"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     modules_to_save=["DenseReluDense.gate", "DenseReluDense.experts.0","DenseReluDense.experts.1","DenseReluDense.experts.2","mask_generator.L","mask_generator.l1","mask_generator.l2","mask_generator.l3"]
    #     modules_to_save=["DenseReluDense.gate", "DenseReluDense.experts_0.0","DenseReluDense.experts_0.1","DenseReluDense.experts_0.2","DenseReluDense.experts_1.0","DenseReluDense.experts_1.1","DenseReluDense.experts_1.2", "mask_generator.L","mask_generator.l1","mask_generator.l2","mask_generator.l3"]
    #     )
    # # add LoRA adaptor
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    param_train, param_percent = count_parameters(model)
    logger.info(f"Trainable parameters: {param_train/1e6}M ({param_percent}%).")
    if args.resume:
        model, config, args = load_ckpt(args, model, config)
    model.initial_weights(config)

    

    # model.to(device)
    

    #### Get the data and dataloader
    trainset = build_dataset(is_train=True, args=args)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, num_workers=0, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=False,
                                                drop_last = True)
    batches = [batch for batch in trainloader]
    random.shuffle(batches)

    #### Get the test dataloader
    testset = build_dataset(is_train=False, args=args)
    testloader = {}
    for task in args.test_task:
        if task == 'mmlu':
            subjects = list(subcategories.keys())
            temploader = {subject: torch.utils.data.DataLoader(dataset=testset[task][subject], num_workers=0, pin_memory=True,
                                                        batch_size=args.batch_size, shuffle=False, drop_last = False)
                                                        for subject in subjects}
            testloader[task]=temploader
        else:
            testloader[task] = torch.utils.data.DataLoader(dataset=testset[task], num_workers=0, pin_memory=True,
                                                        batch_size=args.batch_size, shuffle=False, drop_last = False)
 
    
    if args.eval:
        if testset == None:
            logger.error("No testing data are provided for evaluation")
            sys.exit()
        test_stats = evaluate(args = args, model = model, testloader = testloader, device = device)
        print("On average: ")
        for task in args.test_task:
            print('[Task: %s], total testing samples %d: [loss: %f] [score: %f] [compress rate: %f]' %(task.upper(), len(testloader[task].dataset), test_stats['loss'][task], test_stats['score'][task], test_stats['compression rate'][task]))
        save_result(args=args, dir=args.output_dir, test_stats=test_stats)
        sys.exit()

    if trainset == None:
        logger.error("No training data are provided for training")
        sys.exit()
    logger.info("Start training the model...")

    start_time = time.time()
    
    # Training loop
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-5) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    for epoch in range(args.epochs):
        train_stats = train(args=args, model=model, dataloader=trainloader, optimizer=optimizer, device=device, mode='info')
        save_model(args=args, dir=args.output_dir, model=model, config=config, train_stats=train_stats)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {train_stats['loss']}, Compression rates: {train_stats['compression_rate']}")
        if (epoch+1)%1==0:
            test_stats = evaluate(args = args, model = model, testloader = testloader, device = device)
            save_model(args=args, dir=args.output_dir, model=model, config=config, train_stats=train_stats, test_stats=test_stats)
            # print("On average: ")
            for task in args.test_task:
                print('[Task: %s], total testing samples %d: [loss: %f] [score: %f] [compress rate: %f]' %(task.upper(), len(testloader[task].dataset), test_stats['loss'][task], test_stats['score'][task], test_stats['compression rate'][task]))
        # print(f'Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}')
        scheduler.step()
    test_stats = evaluate(args = args, model = model, testloader = testloader, device = device)
    for task in args.test_task:
        print('[Task: %s], total testing samples %d: [score: %f] [compress rate: %f]' %(task.upper(), len(testloader[task].dataset), test_stats['score'][task], test_stats['compression rate'][task]))
    save_model(args=args, model=model, config=config, train_stats=train_stats, test_stats=test_stats)
    input("Stop")
    optimizer = torch.optim.AdamW(get_param_groups(model=model, mode='chan'), lr = 1e-5) 
    for epoch in range(args.epochs):
        train_stats = train(args=args, model=model, dataloader=trainloader, optimizer=optimizer, device=device, mode='chan')
        print(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {train_stats['loss']}, Compression rates: {train_stats['compression_rate']}")
        if epoch%3==0:
            test_stats = evaluate(args = args, model = model, testloader = testloader, device = device)
            print("On average: ")
            for task in args.test_task:
                print('[Task: %s], total testing samples %d: [score: %f] [compress rate: %f]' %(task.upper(), len(testloader[task].dataset), test_stats['score'][task], test_stats['compression rate'][task]))
            save_model(args=args, model=model, config=config, train_stats=train_stats, test_stats=test_stats)
        scheduler.step()
    input("Stop")
    total_time = time.time()-start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if not testset == None:
        test_stats = evaluate(args = args, model = model, testloader = testloader, device = device)
        print("On average: ")
        for task in args.test_task:
            print('[Task: %s], total testing samples %d: [score: %f] [compress rate: %f]' %(task.upper(), len(testloader[task].dataset), test_stats['score'][task], test_stats['compression rate'][task]))
    save_model(args=args, model=model, config=config, train_stats=train_stats, test_stats=test_stats)
    
    # Tear down the process group
    dist.destroy_process_group()



if __name__ == '__main__':
    opts = get_args()
    main(opts)