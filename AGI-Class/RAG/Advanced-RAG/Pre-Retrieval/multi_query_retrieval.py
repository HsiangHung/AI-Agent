from langchain_core.load import dumps, loads
from operator import itemgetter
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import get_a_t_mix_clients, get_ali_clients
#Multi-Query 多路召回
#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 加载文档
loader = TextLoader("./deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 创建文档分割器，并分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embeddings_model)
# 创建检索器
retriever = vectorstore.as_retriever()

# #检索测试
relevant_docs= retriever.invoke('deepseek的应用场景')
print(relevant_docs)
#查看一下检索到的相关文档的数量：
print("检索器检索的文档数量为：",len(relevant_docs))
# #检索测试-完成


#创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 用来并发的生成N个类似的提问
chain = RunnableMap({
    "context": lambda x: relevant_docs,
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("--------------优化前-------------------")
response = chain.invoke({"question": "deepseek的应用场景"})
print("没有多路召回前，大模型生成的回答：",response)
# 以上是没有多路召回时，我们的大模型生成的回答



print("--------------开始优化-------------------")
#方法一：使用langchain的MultiQueryRetriever
#引入日志组件查看llm在原查询的基础上生成的多个查询
# import logging
# logging.basicConfig()
# # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
#
# # MultiQueryRetriever是对查询的优化
# retrieval_from_llm = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=llm
# )
# # retrieval_from_llm.verbose = True
#
# unique_docs = retrieval_from_llm.invoke({"question":'deepseek的应用场景'})
# print(unique_docs)
# print(len(unique_docs))
#
# exit()


#方法二：自定义prompt
# prompt模版
template = """你是一个AI语言模型助手。你的任务是生成5个给定用户问题的不同版本，以从向量中检索相关文档
数据库。通过对用户问题产生多种观点，你的目标是提供帮助用户克服了基于距离的相似性搜索的一些限制。
提供了这些用换行符隔开的可选问题。原始问题: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
print("--------------自定义prompt生成的扩展问题-------------------")
response = generate_queries.invoke({"question":'deepseek的应用场景'})
print(response)


'''接下来我们使用所有的查询语句（假设为n个）去检索向量数据库，
理论上每条查询语句都会检索出4个相关文档，
那么总共可以检索出 n* 4 个相关文档，但是由于这些查询语句含义相近，
而已检索出来的相关文档可能出现重复，
因此我们必须过滤掉重复的相关文档,只保留唯一的文档'''
# @chain
def get_unique_union(documents: list[list]):
    """ 获取检索文档的唯一并集 """
    # 将列表中的列表展开，并将每个 Document 转换为字符串
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 文档去重
    unique_docs = list(set(flattened_docs))
    # 返回去重后的文档列表
    return [loads(doc) for doc in unique_docs]

# 进行检索
'''retriever.map() 的作用是对输入的查询列表中的每个查询分别进行检索操作，
并返回每个查询对应的检索结果列表。
假设 generate_queries 生成了以下查询列表：["deepseek的应用场景", "deepseek的使用方法", 
"deepseek的优势"]
retriever.map() 对每个查询进行检索，假设检索结果如下：
对于查询 "deepseek的应用场景"，检索到文档列表 docs1,docs2。
对于查询 "deepseek的使用方法"，检索到文档列表 docs2。
对于查询 "deepseek的优势"，检索到文档列表 docs3。
因此，retriever.map() 的输出将是：[docs1,docs2, docs2, docs3]'''
question = "deepseek的应用场景"
retrieval_chain = generate_queries | retriever.map() | RunnableLambda(get_unique_union)
docs = retrieval_chain.invoke({"question":question})
# 去除重复后的知识库数量
print(docs)
print(len(docs))

print("--------------优化后-------------------")
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain, 
    "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)


question = "deepseek的应用场景"
response = final_rag_chain.invoke({"question":question})
print(response)