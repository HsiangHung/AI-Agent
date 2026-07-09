#
#  Implement parent-child-indexing optimization in RAG.
#   * 'ParentDocumentRetriever' as retriever
#   * parent chunk_size=1024, child chunk_size=256
#   * docstore in InMemoryStore
#   * doc context on parents, indexing child vectors.
# 
#
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.stores import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

#父子索引示例代码

# #获得访问大模型和嵌入模型客户端
from langchain_ollama import ChatOllama, OllamaEmbeddings
client = ChatOllama(model="qwen3.5:9b")
embeddings_model = OllamaEmbeddings(model="bge-m3:567m")

# 加载数据
loader = TextLoader("/Users/hhung/Desktop/agi_class/RAG/data/deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

print(len(docs))

# 查看长度
print(f"文章的长度：{len(docs[0].page_content)}")

# exit(0)

# 子块是父块内容的子集
#创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# potential improvement:
# 1. introduce separation tag in parent splitter
# 2. semantic splitting in child splitter

#创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function = embeddings_model,
    # persist_directory="./chroma_parentchild", # without persist_direct, Chroma stores in memory
)

# 创建内存存储对象
store = InMemoryStore()

#创建父子文档检索器，帮我们通过检索子块，返回父文档块
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store, # 文档存储对象
    child_splitter=child_splitter, # 子文档分割器，子文档存储到向量数据库
    parent_splitter=parent_splitter,# 主文档分割器，主文档存储到内存中
    search_kwargs={"k": 5},  # topK = 5,相似度最高的子文档块
)


#添加文档集
retriever.add_documents(docs)

print(f"主文块的数量：{len(list(store.yield_keys()))}")

# # 测试 - 相似性搜索
# '''这里我们通过向量数据库的similarity_search方法搜索出来的是与用户问题相关的子文档块的内容，
# 下面我们使用检索器的get_relevant_documents的方法来对这个问题进行检索，
# 它会返回该子文档块所属的主文档块的全部内容： '''
print("------------similarity_search------------------------")
sub_docs = vectorstore.similarity_search("deepseek的应用场景", k=2)
# print(sub_docs[0].page_content)
for i, doc in enumerate(sub_docs):
    print(i, doc.page_content)
    print()
# print([doc.page_content for doc in sub_docs])

print("------------get_relevant_documents-----------通过子找父-------------")
retrieved_docs = retriever.invoke("deepseek的应用场景")
# print(retrieved_docs[0].page_content)
print(len(retrieved_docs))
for i, doc in enumerate(retrieved_docs):
    print(i, doc.page_content)
    print()
# print([doc.page_content for doc in retrieved_docs])

# 测试 - 相似性搜索 - 完成
# exit()


# 创建prompt模板

template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

#创建chain
chain = RunnableParallel({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | client | StrOutputParser()

print("------------模型回复------------------------")

response = chain.invoke({"question": "deepseek的应用场景"})
print(response)
