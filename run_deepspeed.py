import argparse
import gc
import multiprocessing as mp
import os
import json

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from accelerate import init_empty_weights
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
                          BloomForCausalLM, OPTForCausalLM, LlamaForCausalLM,
                        )
from transformers.deepspeed import HfDeepSpeedConfig
from packaging import version
import pandas as pd
from huggingface_hub import login
login(token="TOKEN")



from src.utils import *
from src.load_model import *
from src.model_utils import *
from src.timer import *

TASKS = ['abstract_algebra']
# TASKS = [
#         'abstract_algebra',
#         'anatomy',
#         'astronomy',
#         'business_ethics',
#         'clinical_knowledge',
#         'college_biology',
#         'college_chemistry',
#         'college_computer_science',
#         'college_mathematics',
#         'college_medicine',
#         'college_physics',
#         'computer_security',
#         'conceptual_physics',
#         'econometrics',
#         'electrical_engineering',
#         'elementary_mathematics',
#         'formal_logic',
#         'global_facts',
#         'high_school_biology',
#         'high_school_chemistry',
#         'high_school_computer_science',
#         'high_school_european_history',
#         'high_school_geography',
#         'high_school_government_and_politics',
#         'high_school_macroeconomics',
#         'high_school_mathematics',
#         'high_school_microeconomics',
#         'high_school_physics',
#         'high_school_psychology',
#         'high_school_statistics',
#         'high_school_us_history',
#         'high_school_world_history',
#         'human_aging',
#         'human_sexuality',
#         'international_law',
#         'jurisprudence',
#         'logical_fallacies',
#         'machine_learning',
#         'management',
#         'marketing',
#         'medical_genetics',
#         'miscellaneous',
#         'moral_disputes',
#         'moral_scenarios',
#         'nutrition',
#         'philosophy',
#         'prehistory',
#         'professional_accounting',
#         'professional_law',
#         'professional_medicine',
#         'professional_psychology',
#         'public_relations',
#         'security_studies', 
#         'sociology',
#         'us_foreign_policy',
#         'virology',
#         'world_religions']

choices = ["A", "B", "C", "D"]




def run_generation(
    model_name,
    batch_size,
    prompt_len,
    gen_len,
    cpu_offload,
    disk_offload,
    offload_dir,
    num_nodes,
    num_gpus_per_node,
    dummy,
    output_file,
    verbose,
    kv_offload,
    quant_bits,
    quant_group_size,
    pin_kv_cache,
    async_kv_offload,
    loops,
    args
):
    # Load tokenizer
    config = get_model_config(model_name)    

    tokenizer = get_tokenizer(model_name, config)

    if dummy:
        filename = os.path.join(
            offload_dir, f"{model_name.replace('/', '-')}-hf-weights/"
        )
        if not os.path.exists(filename):
            print("create dummy weights")
            with init_empty_weights():
                if config.model_type == 'opt':
                    model = OPTForCausalLM(config)
                elif config.model_type in ["bloom", "bloom-7b1"]:
                    model = BloomForCausalLM(config)
                elif config.model_type == "llama":
                    model = LlamaForCausalLM(config)
                else:
                    raise ValueError(f"Unexpected model type: {config.model_type}")                    
            model.save_pretrained(
                filename, state_dict=meta_to_cpu(model.state_dict(), torch.float16)
            )
        dummy_weights = filename
    else:
        dummy_weights = None

    print("load model")
    with torch.no_grad():
        model = get_ds_model(
            model_name,
            cpu_offload,
            disk_offload,
            offload_dir,
            dummy_weights,
            quant_bits,
            quant_group_size,
            args
        )

    # Run generation
    execute_gen_len = gen_len

    def _batch_encode(prompts):
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=prompt_len)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens


    if kv_offload:
        model.set_kv_cache_offload(True, gen_len, pin_kv_cache, async_kv_offload)

    # print(model, model.config)


    add_model_hooks(model)

    def set_model_stage(model, stage):
        model.stage = stage

    # Run
    
    records = []
    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    # global prefill_timings
    # prefill_timings = []
    # global timer
    # timer = timers("generate-forward")
    run_results = {}
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)[0:16]
        
        for i in range(test_df.shape[0]):
            k = args.ntrain
            if args.data_dir == 'data/':
                prompt_end = format_example(test_df, i, include_answer=False)
            else:
                prompt_end = format_example_rummlu(test_df, i, include_answer=False)
            
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(prompt) + 1> args.prompt_len: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})
        end_time = time.time()
        print(records)
        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records], batch_size = args.batch_size, generate_kwargs = generate_kwargs, prompt_len=prompt_len)
        #costs = timers("generate-forward").costs
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}

    if args.local_rank != 0:
        return

    def remove_model_hooks(module):
        if hasattr(module, "__start_time_hook_handle__"):
            module.__start_time_hook_handle__.remove()
            del module.__start_time_hook_handle__
        if hasattr(module, "__end_time_hook_handle__"):
            module.__end_time_hook_handle__.remove()
            del module.__end_time_hook_handle__
        if hasattr(module, "stage"):
            del module.stage
        if hasattr(module, "__duration__"):
            del module.__duration__

    # Log output
    # print(f"Summary:")
    # print(f"costs = {costs}, prefill_timings = {prefill_timings}")
    # total_latency = costs[-1]
    # prefill_latency = prefill_timings[-1]
    remove_model_hooks(model)

    # prefill_throughput = batch_size * prompt_len / prefill_latency
    # decode_latency = total_latency - prefill_latency
    # decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    # num_generated_tokens = batch_size * gen_len
    # total_throughput = num_generated_tokens / total_latency
    total_time = "%.2f"% (end_time - start_time)
    max_mem_allocated = get_accelerator().max_memory_allocated(torch.device("cuda"))
    max_mem_reserved = get_accelerator().max_memory_reserved(torch.device("cuda"))
    
    
    print(f"The total time is: {total_time}")
    print(f"The max mem allocated is: {max_mem_allocated}")
    print(f"The max mem reserved is: {max_mem_reserved}")
    
    output_filename='dummy'
    
    if args.data_dir == 'data/':
        output_filename = 'mmlu_result/'+ output_filename  + "_total_time_" + total_time + "_max_mem_allocated_" + str(max_mem_allocated) + \
    "_max_mem_reserved_" + str(max_mem_reserved) + ".json"
    else:
        output_filename = 'rummlu_result/'+ output_filename + "_total_time_" + total_time + "_max_mem_allocated_" + str(max_mem_allocated) + \
    "_max_mem_reserved_" + str(max_mem_reserved) + ".json"
    
    
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
        
    compute_metric(output_filename)

#     if verbose >= 2:
#         outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#         show_str = "Outputs:\n" + 70 * "-" + "\n"
#         for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
#             show_str += f"{i}: {outputs[i]}\n"
#             show_str += 70 * "-" + "\n"
#         print(show_str)

#         # Check lengths
#         input_lens = [len(x) for x in input_tokens.input_ids]
#         output_lens = [len(x) for x in output_ids]
#         assert all(x == prompt_len for x in input_lens)
#         assert all(x == prompt_len + execute_gen_len for x in output_lens)

    # if output_file == "auto":
    #     filename = (
    #         get_filename(
    #             model_name,
    #             batch_size,
    #             prompt_len,
    #             gen_len,
    #             cpu_offload,
    #             disk_offload,
    #             num_nodes,
    #             num_gpus_per_node,
    #             kv_offload,
    #             quant_bits != 16,
    #         )
    #         + ".log"
    #     )
    # else:
    #     filename = output_file

    # cache_size = cache_bytes(config, batch_size, prompt_len + gen_len)
    # hidden_size = hidden_bytes(config, batch_size, prompt_len + gen_len)
    # log_str = write_benchmark_log(
    #     filename,
    #     model_bytes(config),
    #     cache_size,
    #     hidden_size,
    #     gpu_peak_mem,
    #     prefill_latency,
    #     prefill_throughput,
    #     decode_latency,
    #     decode_throughput,
    #     total_latency,
    #     total_throughput,
    # )
    # if verbose >= 1:
    #     print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="model name or path; currently only supports OPT and BLOOM models")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights for benchmark purposes.")
    parser.add_argument("--loops", type=int, default=3,  help="Number of token generation iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen-len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    parser.add_argument("--pin-memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")
    parser.add_argument("--log-file", type=str, default="auto", help="log file name")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
    parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()

    deepspeed.init_distributed()    
    num_gpus_per_node = get_accelerator().device_count()
    num_nodes = dist.get_world_size() // num_gpus_per_node


    run_generation(
        args.model,
        args.batch_size,
        args.prompt_len,
        args.gen_len,
        args.cpu_offload,
        args.disk_offload,
        os.path.abspath(os.path.expanduser(args.offload_dir)),
        num_nodes,
        num_gpus_per_node,
        args.dummy,
        args.log_file,
        args.verbose,
        args.kv_offload,
        args.quant_bits,
        args.quant_group_size,
        args.pin_kv_cache,
        args.async_kv_offload,
        args.loops,
        args
    )
