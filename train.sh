#!/bin/bash

upload_user="meta-llama"
model_type="Meta-Llama-3.1-8B-Instruct"
padding_side="right"
max_length=2048
lr=3e-5
max_grad_norm=1.0
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=32
tpu_num_cores=32
save_steps=1000

PJRT_DEVICE=TPU python ~/llm-fine-tune-hf-tpu/main.py mode=train \
    upload_user=$upload_user \
    model_type=$model_type \
    padding_side=$padding_side \
    max_length=$max_length \
    lr=$lr \
    max_grad_norm=$max_grad_norm \
    per_device_train_batch_size=$per_device_train_batch_size \
    per_device_eval_batch_size=$per_device_eval_batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    tpu_num_cores=$tpu_num_cores \
    save_steps=$save_steps 
