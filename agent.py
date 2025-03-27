import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from langchain.agents import AgentExecutor
import jsonlines
from langchain.agents import create_openai_functions_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from tqdm import tqdm
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class LangchainDeepSeek:
    def __init__(self, model_name="deepseek-chat", keys_path=None):
        self.model_name = model_name
        self.keys = self._load_keys(keys_path) if keys_path else []
        self.current_key_index = 0

        # 设置初始API密钥
        if self.keys:
            os.environ["DEEPSEEK_API_KEY"] = self.keys[self.current_key_index]

        # 创建agent模型和提示模板
        self.model = ChatDeepSeek(model=self.model_name, temperature=0.7)
        # self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.prompt = '''你是一个有用的助手，可以同时使用以下工具：\n1. medical_document_retriever - 从医学文档中查找专业医疗信息\n2. tavily_search_results_json - 从互联网中查找最新的通用信息\n- 请同时使用两个工具获取更全面的答案'''
        self.tools = self._create_tools()
        self.agent_executor = create_react_agent(model=self.model,tools=self.tools,prompt=self.prompt,debug=True)
        # 创建处理链
        # self.chain = self.prompt | self.model | StrOutputParser()

    def _create_tools(self):
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()

        retrieval_tool = create_retriever_tool(
            retriever,
            "medical_document_retriever",
            "A tool for retrieving information from the medical document"
        )
        search_tool = TavilySearchResults(max_results=4)
        # retrieval_tool is a tool for retrieving information from the medical document
        # search_tool is a tool for searching information from the internet
        tools = [retrieval_tool, search_tool]
        return tools

    def _load_keys(self, keys_path):
        """从文件加载API密钥"""
        keys = []
        with open(keys_path, 'r') as f:
            for line in f:
                key = line.strip()
                if key:
                    keys.append(key)
        return keys

    def _rotate_key(self):
        """轮换到下一个API密钥"""
        if not self.keys:
            return

        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        os.environ["DEEPSEEK_API_KEY"] = self.keys[self.current_key_index]
        # 更新模型以使用新的API密钥
        self.model = ChatDeepSeek(model=self.model_name)
        # 重新创建处理链
        self.chain = self.prompt | self.model | StrOutputParser()

    def __call__(self, message):
        """处理消息并返回响应"""
        if message is None or message == "":
            return "Your input is empty."

        max_attempts = min(len(self.keys), 5) if self.keys else 1
        attempts = 0

        while attempts < max_attempts:
            try:
                response = self.agent_executor.invoke({"messages": [("user", message)]})
                return response["messages"][-1].content
            except Exception as e:
                print(f"Error with key {self.current_key_index}: {e}")
                attempts += 1
                if attempts < max_attempts:
                    self._rotate_key()
                else:
                    return f"Failed after {attempts} attempts. Last error: {e}"


def langchain_datagen(args):
    """使用LangChain处理数据生成"""
    # 初始化LangChain模型
    lds = LangchainDeepSeek(model_name=args.model_name, keys_path=args.keys_path)

    def process_item(item):
        """处理单个数据项"""
        item["model_answer"] = lds(item["query"])
        return item

    output_path = args.output_path
    input_path = args.input_path

    # 收集已处理项目的ID
    processed_ids = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path, "r") as f:
            for item in f:
                processed_ids.add(item.get("id", None))

    # 收集未处理的项目
    items_to_process = []
    with jsonlines.open(input_path, "r") as reader:
        for item in reader:
            item_id = item.get("id", None)
            if item_id is not None and item_id in processed_ids:
                continue
            items_to_process.append(item)

    # 多线程并行处理
    with jsonlines.open(
            output_path, "a" if os.path.exists(output_path) else "w"
    ) as writer:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_item, item): item for item in items_to_process
            }

            # 使用tqdm显示进度
            for future in tqdm(
                    futures, total=len(items_to_process), desc="处理项目中"
            ):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(
                        f"处理项目时出错: {futures[future]['query']}. 错误: {e}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LangChain并发处理JSONL文件。")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-chat",
        help="要使用的DeepSeek模型名称。",
    )
    parser.add_argument(
        "--keys_path",
        type=str,
        required=True,
        help="DeepSeek API密钥文件路径。",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="输入JSONL文件的路径。"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出JSONL文件的路径。"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="并发处理的最大工作线程数。",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com/v1",
        help="API基础URL。",
    )
    # parser.add_argument(
    #     "--database_path",
    #     type=str,
    #     required=True,
    #     help="Path to the database file"
    # )

    args = parser.parse_args()
    print(f"Using url: {args.base_url}")
    os.environ["DEEPSEEK_API_BASE"] = args.base_url
    os.environ["TAVILY_API_KEY"] = "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"
    os.environ["OPENAI_API_KEY"] = "sk-8bWHFZhLVSPyeXoO6f0327Ee96A34a1dB158Ad85174eE5A0"
    os.environ["OPENAI_BASE_URL"] = "https://apix.ai-gaochao.cn/v1"
    langchain_datagen(args)