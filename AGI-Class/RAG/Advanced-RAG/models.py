import os
import dashscope
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
embeddings_model = OllamaEmbeddings(
    model="bge-m3:567m"
)


from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
def get_ali_rerank(top_n=3):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    基于阿里云的 DashScope 大模型 实现，通过语义理解能力对文档与查询的匹配度进行二次评估。
    相比传统检索，语义重排序会增加一定延迟，适合对精度要求高的场景。
    :return: 阿里通义千问嵌入模型的实例

    NOTE here I used "gte-rerank", not "gte-rerank-v2" model. The reason is Based on 
    Alibaba Cloud's official documentation, gte-rerank-v2 is currently only supported 
    in the Mainland China (Beijing) region. An International API key and routing your traffic to the International (Singapore)
    endpoint (dashscope-intl.aliyuncs.com), the server cannot find the v2 model.
    '''
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    return DashScopeRerank(
        model="gte-rerank",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        top_n=top_n
    )
