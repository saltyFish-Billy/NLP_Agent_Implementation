# LLMs as a knowledgeable doctor

This is an LLM agent designed to answer question from 2021 national exam.

The designed agent approach reached 90/100 accuracy answering questions.

# Dependencies

To run the LLM agent,  a deepseek api key is required

you need to put it in the deepseekkey.txt

run following cammand to install required dependencies:
```
pip install retrying, openai, urllib3==1.25.11, tqdm, jsonlines, langchain
```

# Agent Implementation

The agent will automatically use two tools:

- medical_document_retriever - search from local dataset

- tavily_search_results_json - search from web pages

to use the medical_document_retriever, embeddings dataset are built using BGE embedding model in advance

to use the tavily search tool, you need aquire TAVILY_API_KEY from offical website.

# run

use bash.exe to excute the provided scripts in task1 folder.

- first run 1.run_prepare_data.sh to generate question with cot prompting technique(where you could modify based on your own interest).
- second run 2.run_gpt_datagen_multithread.sh to generate agent answers, which will be put in the data folder.
- last run 3.scorer.sh to generate accuracy matrix to evaluate the final result.

# result

[最佳选择题]准确率：0.914  题目总数：35
[配伍选择题]准确率：0.956  题目总数：45
[综合分析选择题]准确率：0.750  题目总数：12
[多项选择题]准确率：0.750  题目总数：8
总分：90  / 满分：100