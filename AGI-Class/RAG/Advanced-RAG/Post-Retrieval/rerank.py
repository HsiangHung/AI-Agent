from langchain_core.documents import Document

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from models import get_ali_rerank

# 初始化阿里重排序模型
reranker = get_ali_rerank()

# ------------------------------------------------
# 示例1：直接对字符串列表进行重排序

query = "孕妇感冒了怎么办"
#  待排序的文档列表（字符串格式）
documents = [
    "感冒应该吃999感冒灵",
    "高血压患者感冒了吃什么",
    "感冒了可以吃感康，但是孕妇禁用"
]

# 使用重排序模型计算文档与查询的相关性得分
# 返回：每个文档的得分列表（数值越高表示相关性越强）
# rerank--一种处理模式，特点：轻量级快速评分
scores = reranker.rerank(documents, query)
# print(scores)
print("示例1：直接对字符串列表进行重排序")
for i, doc in enumerate(scores):
    print(i, doc["relevance_score"], documents[doc["index"]])
print()

# ------------------------------------------------
# 示例2：处理包含元数据的Document对象
documents = [
    Document(
        page_content="感冒应该吃999感冒灵",
        metadata={"source": "999感冒灵"},
    ),
    Document(
        page_content="高血压患者感冒了吃什么",
        metadata={"source": "高血压患者"},
    ),
    Document(
        page_content="感冒了可以吃感康，但是孕妇禁用",
        metadata={"source": "感康"},
    ),
]
# 返回：按相关性降序排列的新文档列表

# 输出：按相关性排序的新文档列表 特点：保留原始文档结构和元数据
scores = reranker.compress_documents(documents, query)
print(scores)
print("示例2：处理包含元数据的Document对象")
for i, doc in enumerate(scores):
    print(
        i,
        doc.metadata["source"],
        doc.metadata["relevance_score"],
    )
print()
