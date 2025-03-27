#!/bin/bash

# Set the path to the Python script
#python_script="../langchain_datagen_multithread.py"
python_script="../agent.py"
# set parameter
# keys_path="../gpt3keys.txt"
keys_path="../deepseekkeys.txt"
input_path="./data/2.exam_prepared.jsonl"
output_path="./data/3.exam_aftgpt.jsonl"
#database_path="./data/"
# base_url="https://apix.ai-gaochao.cn/v1"

base_url="https://api.deepseek.com/v1"
max_workers=10

#python "$python_script" --use_agent --keys_path "$keys_path" --input_path "$input_path" --output_path "$output_path" --max_workers $max_workers --base_url $base_url
python "$python_script" --keys_path "$keys_path" --input_path "$input_path" --output_path "$output_path" --max_workers $max_workers --base_url $base_url
# python ../langchain_datagen_multithread.py --keys_path ../gpt3keys.txt --input_path ./data/2.exam_prepared.jsonl --output_path ./data/3.exam_aftgpt.jsonl --max_workers 10
