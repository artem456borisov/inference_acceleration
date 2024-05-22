from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
                          BloomForCausalLM, OPTForCausalLM, LlamaForCausalLM,
                        )
from transformers.deepspeed import HfDeepSpeedConfig

from .model_utils import (GB, add_model_hooks, cache_bytes,
                   get_filename, get_quant_config, hidden_bytes, meta_to_cpu,
                   model_bytes, write_benchmark_log)

import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
import torch
import gc
import multiprocessing as mp
import os

def get_tokenizer(model_name, config):
    if config.model_type == "opt":
        # opt175b is not available on HF (at this time),
        # so as a hack we use opt66b which has similar tokenizer. 
        tokenizer = AutoTokenizer.from_pretrained(
            model_name.replace("175b", "66b"), 
            padding_side="left" 
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model_config(model_name):
    if "175b" in model_name:
        config = AutoConfig.from_pretrained("facebook/opt-66b")
        config.hidden_size = 12288
        config.word_embed_proj_dim = 12288
        config.ffn_dim = 12288 * 4
        config.num_attention_heads = 96
        config.num_hidden_layers = 96
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if 'bloom' in model_name:
        config.model_type = 'bloom'

    return config


def get_ds_model(
    model_name,
    cpu_offload,
    disk_offload,
    offload_dir,
    dummy_weights,
    bits,
    group_size,
    args
):

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    pin_memory = bool(args.pin_memory)

    if getattr(config, 'torch_dtype', None) is None:
        dtype = torch.float16
    else:
        dtype = config.torch_dtype

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, # 0, 
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": args.batch_size,
        "wall_clock_breakdown": False,
    }

    if bits == 4:
        quant_config = get_quant_config(config, bits=bits, group_size=group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=9 * GB if config.model_type == 'bloom' else 2 * GB,
        )
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        }

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    get_accelerator().empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    model = model.eval()


    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f"model.config = {model.config}")

    return model