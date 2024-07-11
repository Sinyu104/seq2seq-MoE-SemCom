import torch
import sys
import os
import json
from pathlib import Path
import datetime
from loguru import logger
from model import T5SC_model
from dataset import SST, IWSLT, SQuAD, MMLU, Glue_mrpc, Glue_qqp
from torch.utils.data import Dataset
from evaluate import load
from torch.cuda.amp import GradScaler





def get_model(args, config):
    print(f"Creating model: {args.model}")
    model = T5SC_model.from_pretrained(pretrained_model_name_or_path=args.pretrain_model, config=config)
    
    # Stop gradient for pre-trained model
    # for name, param in model.named_parameters():
    #     if not ('FSM' in name or 'noise_func' in name):
    #         param.requires_grad_(False)
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
                for task in args.train_task:
                    if task.lower() == 'sen':
                        print(f"Building dataset for sentiment analysis task...")
                        self.SST = SST(train=is_train)
                    elif task.lower() == 'trans':
                        print(f"Building dataset for translation task...")
                        self.IWSLT = IWSLT(train=is_train)
                    elif task.lower() == 'qa':
                        print(f"Building dataset for QA task...")
                        self.SQuAD = SQuAD(train=is_train)
                    elif task.lower() == 'mmlu':
                        print(f"Building dataset for MMLU task...")
                        self.MMLU = MMLU(train=is_train)
                    elif task.lower() == 'glue_mrpc':
                        print(f"Building dataset for Glue/mrpc task...")
                        self.Glue_mrpc = Glue_mrpc(train=is_train)
                    elif task.lower() == 'glue_qqp':
                        print(f"Building dataset for Glue/qqp task...")
                        self.Glue_qqp = Glue_qqp(train=is_train)
                    else:
                        raise NotImplementedError

                self.length = [('-', 0)]
                for task in args.train_task:
                    if task.lower() == 'sen':
                        self.SST_len = (len(self.SST)//args.batch_size)*args.batch_size
                        self.length.append(('sen', self.length[-1][1]+self.SST_len)) 
                    elif task.lower() == 'trans':
                        self.IWSLT_len = (len(self.IWSLT)//args.batch_size)*args.batch_size
                        self.length.append(('trans', self.length[-1][1]+self.IWSLT_len)) 
                    elif task.lower() == 'qa':
                        self.SQuAD_len = (len(self.SQuAD)//args.batch_size)*args.batch_size
                        self.length.append(('qa', self.length[-1][1]+self.SQuAD_len))
                    elif task.lower() == 'mmlu':
                        self.mmlu_len = (len(self.MMLU)//args.batch_size)*args.batch_size
                        self.length.append(('mmlu', self.length[-1][1]+self.mmlu_len))
                    elif task.lower() == 'glue_mrpc':
                        self.glue_mrpc_len = (len(self.Glue_mrpc)//args.batch_size)*args.batch_size
                        self.length.append(('glue_mrpc', self.length[-1][1]+self.glue_mrpc_len))
                    elif task.lower() == 'glue_qqp':
                        self.glue_qqp_len = (len(self.Glue_qqp)//args.batch_size)*args.batch_size
                        self.length.append(('glue_qqp', self.length[-1][1]+self.glue_qqp_len))
                    else:
                        raise NotImplementedError
                
                
            def __len__(self):
                
                return self.length[-1][1]

            def __getitem__(self, idx):
                for ta in range(1,len(self.length)):
                    if self.length[ta][1]>idx:
                        temp_idx = ta
                        break
                if self.length[ta][0] == 'sen':
                    return self.SST[idx-self.length[ta-1][1]]
                elif self.length[ta][0] == 'trans':
                    return self.IWSLT[idx-self.length[ta-1][1]]
                elif self.length[ta][0] == 'qa':
                    return self.SQuAD[idx-self.length[ta-1][1]]
                elif self.length[ta][0] == 'mmlu':
                    return self.MMLU[idx-self.length[ta-1][1]]
                elif self.length[ta][0] == 'glue_mrpc':
                    return self.Glue_mrpc[idx-self.length[ta-1][1]]
                elif self.length[ta][0] == 'glue_qqp':
                    return self.Glue_qqp[idx-self.length[ta-1][1]]
                else:
                    raise NotImplementedError
        return CombinedDataset(is_train, args)
    else:
        if args.test_task == '':
            logger.warning("No testing tasks are specified!")
            return None
        SeperatedDataset={}
        for task in args.test_task:
            if task.lower() == 'sen':
                SeperatedDataset['sen']=SST(train=is_train)
            elif task.lower() == 'trans':
                SeperatedDataset['trans']=IWSLT(train=is_train)
            elif task.lower() == 'qa':
                SeperatedDataset['qa']=SQuAD(train=is_train)
            elif task.lower() == 'mmlu':
                SeperatedDataset['mmlu']=MMLU(train=is_train)
            elif task.lower() == 'glue_mrpc':
                SeperatedDataset['glue_mrpc']=Glue_mrpc(train=is_train)
            elif task.lower() == 'glue_qqp':
                SeperatedDataset['glue_qqp']=Glue_qqp(train=is_train)
            else:
                raise NotImplementedError
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
            params.append(param)
            # if 'mask_generator.L' in name or 'mask_generator.l1' in name or 'mask_generator.l2' in name or 'mask_generator.l3' in name or 'lora' in name:
            #     params.append(param)
            # else:
            #     pass
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
