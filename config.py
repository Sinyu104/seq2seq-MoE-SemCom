from transformers import AutoConfig 

def T5SC_config(developer="google", model="flan-t5", mode="small"):
    config = AutoConfig.from_pretrained(f"{developer}/{model}-{mode}")
    config.num_infoFSM=1
    config.num_chanFSM=0
    config.is_channel_disable = False
    config.weight = 20
    config.weight_decay=0.95
    config.distortion = 0.0
    config.num_token=256
    return config
