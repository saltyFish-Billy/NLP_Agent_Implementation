import json
import argparse
# one-shot prompt
# prompt = '''
# 下面是一道{question_type}，请先详细分析问题，最后给出选项。
# {question}
# {option}
# '''

# few-shot prompt
# prompt = '''
# 下面是一道{question_type}，请先详细分析问题，最后给出正确的选项（如'A','B','C','D',或者'E'）。
# {question}
# {option}
# '''

# CoT prompt
# prompt = '''
# 下面是一道{question_type}，请先详细分析问题，最后给出正确的选项:
# **题目：**
# {question}
# **选项：**
# {option}
# **思考步骤要求：**
# 1. 先理解题目背景和关键违规行为。
# 2. 逐个分析选项，判断其是否符合题目描述。
# 3. 找出选项中与题目描述矛盾或错误的结论。
# 4. 最终仅输出答案选项（如 `A`、`B`、`C`、`D` 或 `E`）
# '''

# improved prompt for CoT
# prompt = '''
# 下面是一道{question_type}，请先详细分析问题，最后给出正确的选项:
# **题目：**
# {question}
# **选项：**
# {option}
# **思考步骤要求：**
# 1. 确定根据题目类型确定是多项选择题还是单项选择题。
# 2. 先理解题目背景和关键违规行为。
# 3. 逐个分析选项，判断其是否符合题目描述。
# 4. 找出选项中与题目描述矛盾或错误的结论。
# 5. 最终仅输出答案选项，如果是单项选择题，只输出最有的答案选项（如'A',不包含选项具体内容）如果是多项选择题，把所有满足题意的选项融合输出（如'ABC'，'BCE'，不包含具体选项内容）。
# '''

# agent prompt
prompt = '''
下面是一道{question_type}，请先详细分析问题，最后给出正确的选项:
**题目：**
{question}
**选项：**
{option}
**思考步骤要求：**
1. 确定根据题目类型确定是多项选择题还是单项选择题。
2. 先理解题目背景和关键违规行为。
3. 逐个分析选项，判断其是否符合题目描述。
4. 找出选项中与题目描述矛盾或错误的结论。
5. 最终仅输出答案选项，如果是单项选择题，只输出最有的答案选项（如'A',不包含选项具体内容）如果是多项选择题，把所有满足题意的选项融合输出（如'ABC'，'BCE'，不包含具体选项内容）。
'''

def generate_query(data):
    chatgpt_query = prompt
    question = data['question']
    option = '\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
    chatgpt_query = chatgpt_query.format_map({'question':question,'option':option,'question_type':data['question_type']})
    return chatgpt_query


def Prepare_data(args):
    data = []
    # 读取上传的JSON文件
    with open(args.input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"len:{len(data)}")
    # 根据要求转换
    jsonl_data = []


    for id, item in enumerate(data):
        jsonl_data.append(
            {
                "id":id,
                "query": generate_query(item),
                "model_answer": "",
                "question_type": item['question_type'],
                "groundtruth": item['answer']
            }
        )

    # 将转换后的数据保存为JSONL文件
    with open(args.output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Prepare finished, output to '{args.output_path}'")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for OpenAIGPT generation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()
    Prepare_data(args)
