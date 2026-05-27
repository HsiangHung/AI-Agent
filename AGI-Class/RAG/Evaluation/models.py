import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

qwen_model = ChatOpenAI(
    model="qwen-max",        # Equivalent to specifying "qwen-max"
    temperature=0.2,        # Controls creativity
    max_tokens=2000,        # Maximum output length
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)


qwen_plus_model = ChatOpenAI(
    model="qwen-plus-2025-09-11",        # Equivalent to specifying "qwen-max"
    temperature=0.2,        # Controls creativity
    max_tokens=2000,        # Maximum output length
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

from langchain_ollama import OllamaEmbeddings
embedding_model = OllamaEmbeddings(
    model="bge-m3:567m"
)