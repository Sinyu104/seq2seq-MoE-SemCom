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
from accelerate import init_empty_weights, load_checkpoint_and_dispatch





def get_model(args, config):
    print(f"Creating model: {args.model}")
    # with init_empty_weights():
    #     model = T5SC_model(config)
    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint='modelckpt_2024-09-26_03-30-13/model.pth', device_map="auto", no_split_module_classes=['Block']
    # )
    model = T5SC_model.from_pretrained(pretrained_model_name_or_path=args.pretrain_model, config=config, device_map="auto")
    # Stop gradient for pre-trained model
    # for name, param in model.named_parameters():
    #     if not ('FSM' in name or 'noise_func' in name):
    #         param.requires_grad_(False)
        # if ('gate' in name or 'expert' in name or 'codebook' in name):
        #     param.requires_grad = True
        # else: 
        #     param.requires_grad = False
    return model

def count_parameters(model):
    # for name, param in model.named_parameters():
    #     if 'FSM' in name or 'mask_generator.L' in name or 'mask_generator.l1' in name or 'mask_generator.l2' in name or 'mask_generator.l3' in name :
    #         param.requires_grad_(True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, (trainable_params / total_params) * 100

def load_ckpt(args, model, config=None):
    print(f"Load checkpoint: {args.resume}")
    saved_model_data = torch.load(args.resume)
    config = config if config is not None else (saved_model_data['config'] if 'config' in saved_model_data else AutoConfig.from_pretrained("google/flan-t5-small"))
    # args = saved_model_data['args'] if 'config' in saved_model_data else args
    # new_state_dict = {}
    # for key, value in saved_model_data['model_state_dict'].items():
    #     # Look for keys that match the layers and submodules you want to rename
    #     if "DenseReluDense.wi_0.weight" in key:
    #         # Change the key to include 'experts.expert0'
    #         new_key = key.replace(
    #             "DenseReluDense.wi_0.weight",  # Original part of the key
    #             "DenseReluDense.experts.expert_0.wi_0.weight"  # New part of the key
    #         )
    #         new_state_dict[new_key] = value
    #     elif "DenseReluDense.wi_1.weight" in key:
    #         # Similarly change other parts as needed, for example wi_1
    #         new_key = key.replace(
    #             "DenseReluDense.wi_1.weight",
    #             "DenseReluDense.experts.expert_0.wi_1.weight"
    #         )
    #         new_state_dict[new_key] = value
    #     elif "DenseReluDense.wo.weight" in key:
    #         # Similarly change other parts as needed, for example wi_1
    #         new_key = key.replace(
    #             "DenseReluDense.wo.weight",
    #             "DenseReluDense.experts.expert_0.wo.weight"
    #         )
    #         new_state_dict[new_key] = value
    #     else:
    #         # Keep other keys the same
    #         new_state_dict[key] = value
    # saved_model_data['model_state_dict'] = new_state_dict
    
    model_state_dict = model.state_dict()
    for k in saved_model_data['model_state_dict']:
        if k in model_state_dict:
            if saved_model_data['model_state_dict'][k].shape != model_state_dict[k].shape:
                logger.info(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {saved_model_data['model_state_dict'][k].shape}")
                saved_model_data['model_state_dict'][k] = model_state_dict[k]
                
        else:
            logger.info(f"Dropping parameter {k}")
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
                        case 'copa':
                            self.dataset_list[task] = Copa(train=is_train)
                        case 'arc_easy':
                            self.dataset_list[task] = ARC_easy(train=is_train)
                        case 'hella':
                            self.dataset_list[task] = HellaSwag(train=is_train)
                        case 'winog':
                            self.dataset_list[task] = Winogrande(train=is_train)
                        case 'cosmos':
                            self.dataset_list[task] = Cosmos(train=is_train)
                        case 'agnews':
                            self.dataset_list[task] = Ag_news(train=is_train)
                        case 'commonsense_qa':
                            self.dataset_list[task] = Commonsense_QA(train=is_train)
                        case 'dream':
                            self.dataset_list[task] = Dream(train=is_train)
                        case 'adversarial_qa_dbert':
                            self.dataset_list[task] = Adversarial_QA_dbert(train=is_train)
                        case 'adversarial_qa_dbidaf':
                            self.dataset_list[task] = Adversarial_QA_dbidaf(train=is_train)
                        case 'adversarial_qa_droberta':
                            self.dataset_list[task] = Adversarial_QA_droberta(train=is_train)
                        case 'gigaword':
                            self.dataset_list[task] = Gigaword(train=is_train)
                        case 'wic':
                            self.dataset_list[task] = WiC(train=is_train)
                        case 'common_gen':
                            self.dataset_list[task] = Common_gen(train=is_train)
                        case 'quartz':
                            self.dataset_list[task] = Quartz(train=is_train)
                    

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
                case 'copa':
                    SeperatedDataset[task]=Copa(train=is_train)
                case 'arc_easy':
                    SeperatedDataset[task]=ARC_easy(train=is_train)
                case 'hella':
                    SeperatedDataset[task]=HellaSwag(train=is_train)
                case 'winog':
                    SeperatedDataset[task]=Winogrande(train=is_train)
                case 'cosmos':
                    SeperatedDataset[task]=Cosmos(train=is_train)
                case 'agnews':
                    SeperatedDataset[task]=Ag_news(train=is_train)
                case 'commonsense_qa':
                    SeperatedDataset[task]=Commonsense_QA(train=is_train)
                case 'dream':
                    SeperatedDataset[task]=Dream(train=is_train)
                case 'adversarial_qa_dbert':
                    SeperatedDataset[task]=Adversarial_QA_dbert(train=is_train)
                case 'adversarial_qa_dbidaf':
                    SeperatedDataset[task]=Adversarial_QA_dbidaf(train=is_train)
                case 'adversarial_qa_droberta':
                    SeperatedDataset[task]=Adversarial_QA_droberta(train=is_train)
                case 'gigaword':
                    SeperatedDataset[task]=Gigaword(train=is_train)
                case 'wic':
                    SeperatedDataset[task]=WiC(train=is_train)
                case 'common_gen':
                    SeperatedDataset[task]=Common_gen(train=is_train)
                case 'quartz':
                    SeperatedDataset[task]=Common_gen(train=is_train)

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
        elif task.lower() == 'mnli':
            metrics['mnli'] = load("exact_match")
        elif task.lower() == 'qnli':
            metrics['qnli'] = load("exact_match")
        elif task.lower() == 'boolq':
            metrics['boolq'] = load("exact_match")
        elif task.lower() == 'copa':
            metrics['copa'] = load("exact_match")
        elif task.lower() == 'arc_easy':
            metrics['arc_easy'] = load("exact_match")
        elif task.lower() == 'hella':
            metrics['hella'] = load("exact_match")
        elif task.lower() == 'winog':
            metrics['winog'] = load("exact_match")
        elif task.lower() == 'cosmos':
            metrics['cosmos'] = load("exact_match")
        elif task.lower() == 'agnews':
            metrics['agnews'] = load("exact_match")
        elif task.lower() == 'commonsense_qa':
            metrics['commonsense_qa'] = load("exact_match")
        elif task.lower() == 'dream':
            metrics['dream'] = load("exact_match")
        elif task.lower() == 'adversarial_qa_dbert':
            metrics['adversarial_qa_dbert'] = load("rouge")
        elif task.lower() == 'adversarial_qa_dbidaf':
            metrics['adversarial_qa_dbidaf'] = load("rouge")
        elif task.lower() == 'adversarial_qa_droberta':
            metrics['adversarial_qa_droberta'] = load("rouge")
        elif task.lower() == 'gigaword':
            metrics['gigaword'] = load("rouge")
        elif task.lower() == 'wic':
            metrics['wic'] = load("exact_match")
        elif task.lower() == 'common_gen':
            metrics['common_gen'] = load("sacrebleu")
        elif task.lower() == 'quartz':
            metrics['quartz'] = load("exact_match")
        else:
            raise NotImplementedError
    return metrics

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)



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



