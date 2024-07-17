import torch
import sys
import os
import json
from pathlib import Path
import datetime
from loguru import logger
from model import T5SC_model
from dataset import *
from torch.utils.data import Dataset
from evaluate import load
from torch.cuda.amp import GradScaler





def get_model(args, config):
    print(f"Creating model: {args.model}")
    model = T5SC_model.from_pretrained(pretrained_model_name_or_path=args.pretrain_model, config=config)
    
    # Stop gradient for pre-trained model
    for name, param in model.named_parameters():
        if not ('gate' in name or 'expert' in name):
            param.requires_grad_(False)
    return model

def count_parameters(model):
    # for name, param in model.named_parameters():
    #     if 'mask_generator.L' in name or 'mask_generator.l1' in name or 'mask_generator.l2' in name or 'mask_generator.l3' in name :
    #         param.requires_grad_(True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, (trainable_params / total_params) * 100

def load_ckpt(args, model):
    print(f"Load checkpoint: {args.resume}")
    saved_model_data = torch.load(args.resume)
    config = saved_model_data['config'] if 'config' in saved_model_data else AutoConfig.from_pretrained("google/flan-t5-small")
    # args = saved_model_data['args'] if 'config' in saved_model_data else args
    model.load_state_dict(saved_model_data['model_state_dict'], strict=False)

    return model, config, args

def build_dataset(is_train, args):
    if is_train:
        if args.train_task == '':
            logger.warning("No training tasks are specified!")
            return None
        class CombinedDataset(Dataset):
            def __init__(self, is_train, args):
                self.dataset_list={}
                for task in args.train_task:
                    print("Building dataset for {} task...".format(task))
                    match task:
                        case 'sen':
                            self.dataset_list[task] = SST(train=is_train)
                        case 'trans':
                            self.dataset_list[task] = IWSLT(train=is_train)
                        case 'qa':
                            self.dataset_list[task] = SQuAD(train=is_train)
                        case 'mmlu':
                            self.dataset_list[task] = MMLU(train=is_train)
                        case 'glue_mrpc':
                            self.dataset_list[task] = Glue_mrpc(train=is_train)
                        case 'glue_qqp':
                            self.dataset_list[task] = Glue_qqp(train=is_train)
                        case 'labeled_final':
                            self.dataset_list[task] = labeled_final(train=is_train)
                        case 'anli':
                            self.dataset_list[task] = Anli(train=is_train)
                        case 'mnli':
                            self.dataset_list[task] = Mnli(train=is_train)
                        case 'qnli':
                            self.dataset_list[task] = Qnli(train=is_train)
                        case 'boolq':
                            self.dataset_list[task] = BoolQ(train=is_train)
                    

                self.length = [('-', 0)]
                for task in args.train_task:
                    self.task_len = (len(self.dataset_list[task])//args.batch_size)*args.batch_size
                    self.length.append((task, self.length[-1][1]+self.task_len)) 
                    
                
                
            def __len__(self):
                
                return self.length[-1][1]

            def __getitem__(self, idx):
                for ta in range(1,len(self.length)):
                    if self.length[ta][1]>idx:
                        temp_idx = ta
                        break
                
                return self.dataset_list[self.length[ta][0]][idx-self.length[ta-1][1]]
                
        return CombinedDataset(is_train, args)
    else:
        if args.test_task == '':
            logger.warning("No testing tasks are specified!")
            return None
        SeperatedDataset={}
        for task in args.test_task:
            match task:
                case 'sen':
                    SeperatedDataset[task]=SST(train=is_train)
                case 'trans':
                    SeperatedDataset[task]=IWSLT(train=is_train)
                case 'qa':
                    SeperatedDataset[task]=SQuAD(train=is_train)
                case 'mmlu':
                    SeperatedDataset[task]=MMLU(train=is_train)
                case 'glue_mrpc':
                    SeperatedDataset[task]=Glue_mrpc(train=is_train)
                case 'glue_qqp':
                    SeperatedDataset[task]=Glue_qqp(train=is_train)
                case 'labeled_final':
                    SeperatedDataset[task]=labeled_final(train=is_train)
                case 'anli':
                    SeperatedDataset[task]=Anli(train=is_train)
                case 'mnli':
                    SeperatedDataset[task]=Anli(train=is_train)
                case 'qnli':
                    SeperatedDataset[task]=Qnli(train=is_train)
                case 'boolq':
                    SeperatedDataset[task]=BoolQ(train=is_train)

        return SeperatedDataset

def task_metrics_mapping(args):
    metrics = {}
    for task in args.test_task:
        if task.lower() == 'sen':
            metrics['sen'] = load("exact_match")
        elif task.lower() == 'trans':
            metrics['trans'] = load("sacrebleu")
        elif task.lower() == 'qa':
            metrics['qa'] = load("rouge")
        elif task.lower() == 'mmlu':
            metrics['mmlu'] = load("exact_match")
        elif task.lower() == 'glue_mrpc':
            metrics['glue_mrpc'] = load("exact_match")
        elif task.lower() == 'glue_qqp':
            metrics['glue_qqp'] = load("exact_match")
        elif task.lower() == 'labeled_final':
            metrics['labeled_final'] = load("exact_match")
        elif task.lower() == 'anli':
            metrics['anli'] = load("exact_match")
        elif task.lower() == 'qnli':
            metrics['qnli'] = load("exact_match")
        elif task.lower() == 'boolq':
            metrics['boolq'] = load("exact_match")
        else:
            raise NotImplementedError
    return metrics


def save_model(args, model, config, dir='', train_stats=None, test_stats=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path('modelckpt_'+current_time) if dir=='' else dir
    path_exists_make(output_dir)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'args': args,
    }, f'{output_dir}/model.pth')
    # Write performance as a JSON file
    if not train_stats == None:
        with open(f'{output_dir}/train_stats.json', 'w') as json_file:
            json.dump(train_stats, json_file, indent=4)
    if not test_stats == None:
        with open(f'{output_dir}/test_stats.json', 'w') as json_file:
            json.dump(test_stats, json_file, indent=4)

def save_result(args, dir='',test_stats=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = Path('modelckpt_'+current_time) if dir=='' else dir
    path_exists_make(dir)
    # Write performance as a JSON file
    with open(f'{dir}/test_stats.json', 'w') as json_file:
        json.dump(test_stats, json_file, indent=4)


def path_exists_make(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def get_param_groups(model, mode):
    params = []
    if mode == 'info':
        for name, param in model.named_parameters():
            if 'gate' in name or 'expert' in name:
                params.append(param)
    else:
        for name, param in model.named_parameters():
            if 'FSMs.0' in name:
                pass
            else:
                params.append(param)
    return params


class NativeScaler:
    def __init__(self, scale_factor=2**16, max_norm=None):
        self.scaler = GradScaler(init_scale=scale_factor)
        self.max_norm = max_norm

    def __call__(self, step, loss, optimizer, accumulation_steps=3):
        loss = loss / accumulation_steps # Scale loss
        self.scaler.scale(loss).backward() # Compute gradients
        
        # Clip gradients by norm
        if self.max_norm is not None:
            self.scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.max_norm)
        else:
            self.scaler.unscale_(optimizer)
         # Step the optimizer
        self.scaler.step(optimizer)
        self.scaler.update()

        # Perform the update and reset gradients periodically
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
