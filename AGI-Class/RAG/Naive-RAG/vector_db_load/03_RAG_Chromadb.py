import chromadb

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document


# 安装
# pip install chromadb
# pip install langchain-chroma


# 嵌入模型：
# embeddings = DashScopeEmbeddings(model="text-embedding-v2",)

# Ollama usage:
# https://reference.langchain.com/python/langchain-ollama/embeddings/OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="bge-m3:567m")

# also probably works:
# embeddings_model = OpenAIEmbeddings(
#     api_key=os.getenv("SILICONFLOW_API_KEY"),
#     base_url="https://api.siliconflow.cn/v1",
#     model="Qwen/Qwen3-Embedding-8B",
# )

input_text = "The meaning of life is 42"
vector = embeddings.embed_query(input_text)
print(len(vector))
print(vector[:3])

# exit(0)

# 评分方式
score_measures = [
    "default",  # default = 'l2'
    "cosine",  # 余弦相似度， 1-cos(角度) 0-2
    "l2",  # 欧氏距离，
    "ip"  # 点积， 和 cosine 接近
]

# 创建向量库
db = Chroma(
    collection_name="collection_name",
    embedding_function=embeddings,
    persist_directory="./chroma_db1",
    collection_metadata={"hnsw:space": 'cosine'}
)

# 提供文档
documents = [
    Document(page_content="这个苹果手机很好用"),
    Document(page_content="我国山东地区盛产苹果"),
]

# 添加文档
ids = db.add_documents(documents)
print(ids)
print('-' * 100)

# 检索
results = db.similarity_search_with_score("我想买个手机")
for doc, score in results:
    print(doc.page_content, end='\t')
    print(f"score: {score}")

