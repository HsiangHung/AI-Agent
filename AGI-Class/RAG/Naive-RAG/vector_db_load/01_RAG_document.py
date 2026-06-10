from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, PyMuPDFLoader, TextLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
# from models import *

# 安装
# pip install pymupdf


'''
   文档加载
'''
# 1.文档加载器：网页
'''
web_path = "https://www.news.cn/fortune/20251218/93d4a4212fd34985ad05d178a27c7c75/c.html"

loader = WebBaseLoader(
    web_path=[web_path],
    bs_kwargs=dict(parse_only = bs4.SoupStrainer(class_=("main-left left", "title")))
)
docs = loader.load()
print(docs)  # Document对象
print('-' * 100)
exit()
'''

# 2.文档加载器：txt
'''
loader = TextLoader("../datas/deepseek.txt", encoding="utf-8")
docs = loader.load()
print(docs)
print('-' * 100)
exit()
'''


# 3.文档加载器：docx文件
'''
loader = Docx2txtLoader("../datas/行业.docx")
docs = loader.load()
print(docs)
print('-' * 100)
exit()
'''

# 4.文档加载器: pdf文件
loader = PyMuPDFLoader("/Users/hhung/Desktop/agi_class/RAG/data/Reflexion.pdf")
docs = loader.load()
# print(docs)
# print('-' * 100)

# minerU


'''
    文档切分
'''
#
#  1. RecursiveCharacterTextSplitter：智能递归分割

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    separators=['\n\n', '\n', ' ', '']
)
documents = splitter.split_documents(docs)
for doc in documents:
    print(doc, end='\n --- \n')
print('-' * 100)
print(len(documents))  # 137
exit()


# 2. CharacterTextSplitter：文本按分隔符简单分割
'''
text = "这是第一段。\n\n这是第二段。\n这是第三段。"
splitter = CharacterTextSplitter(
    separator="\n",  # 简单切分
    chunk_size=10,
    chunk_overlap=2
)
chunks = splitter.split_text(text)
print(chunks)
# ['这是第一段。', '这是第二段。\n这是第三段。']
print('-' * 100)
exit()
'''

# 3. MarkdownHeaderTextSplitter: 保留Markdown文档层级结构的分割
markdown = """
# 第一章
## 第一节
这是第一节内容
## 第二节
这是第二节内容
# 第二章
这是第二章内容
"""
headers = [
    ("#", "Header1"),
    ("##", "Header2")
]
# 按headers拆分
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown)
print(chunks)



