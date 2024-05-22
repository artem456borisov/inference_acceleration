#deepspeed --num_gpus 2 run_deepspeed.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --batch_size 2 --prompt_len 4096 --gen-len 1 --quant_bits 4 --data_dir=data_rummlu

deepspeed --num_gpus 2 run_deepspeed.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --batch_size 2 --prompt_len 4096 --gen-len 1 --data_dir='data/'
