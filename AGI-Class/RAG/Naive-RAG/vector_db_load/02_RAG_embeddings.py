from langchain_community.embeddings import DashScopeEmbeddings
import os
from models import *

# 1.初始化 Embeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    # dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 2.使用嵌入模型
# 对单个文本进行嵌入
text = "这是一个测试文本。"
embedding = embeddings.embed_query(text)
print(f"嵌入向量：{embedding}")
print(f"单个文本嵌入维度：{len(embedding)}")
