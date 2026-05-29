# rag_chain.py
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import redis
import hashlib
import os
from vector_store import vector_store
from chat_manager import chat_manager
from config import *

redis_client = redis.from_url(REDIS_URL)


class RAGChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model='qwen-max',
            temperature=0.1
        )
        self.prompt = self._get_finance_prompt()
        self.base_chain = self.prompt | self.llm | StrOutputParser()
        self.rag_chain = chat_manager.wrap_chain(self.base_chain)

    def _get_finance_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """你是专业财务分析师，仅依据提供的文档回答问题。
文档片段：{context}
要求：准确、严谨、简洁，无信息时明确说明"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

    def _get_hybrid_retriever(self, kb_id):
        db = vector_store.get_db(kb_id)
        docs = db.get()["documents"]
        if not docs:
            return db.as_retriever()
        chroma_ret = db.as_retriever(search_kwargs={"k": 5})
        bm25_ret = BM25Retriever.from_texts(docs)
        bm25_ret.k = 5
        return EnsembleRetriever(retrievers=[chroma_ret, bm25_ret], weights=[0.7, 0.3])

    def retrieve(self, kb_id, query):
        cache_key = f"rag:{kb_id}:{hashlib.md5(query.encode()).hexdigest()}"
        if redis_client.exists(cache_key):
            return redis_client.get(cache_key).decode()
        retriever = self._get_hybrid_retriever(kb_id)
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        redis_client.setex(cache_key, CACHE_TTL, context)
        return context

    def chat(self, kb_id, session_id, question):
        """原有方法，仅返回答案生成器（不包含引用）"""
        context = self.retrieve(kb_id, question)
        response = self.rag_chain.stream(
            {"input": question, "context": context, "chat_history": []},
            config={"configurable": {"session_id": session_id}}
        )
        for token in response:
            yield str(token)

    def chat_with_citations(self, kb_id, session_id, question):
        """
        返回 (answer_generator, citations_list)
        - answer_generator: 生成答案 token 的生成器
        - citations_list: 检索到的文档片段列表，每个元素为 dict {content, source, page}
        """
        # 1. 获取检索器并检索文档
        retriever = self._get_hybrid_retriever(kb_id)
        docs = retriever.invoke(question)

        # 2. 构建上下文和引用
        context = "\n".join([d.page_content for d in docs])
        citations = []
        for doc in docs:
            citations.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "未知文件"),
                "page": doc.metadata.get("page", "")
            })

        # 3. 生成答案流
        response = self.rag_chain.stream(
            {"input": question, "context": context, "chat_history": []},
            config={"configurable": {"session_id": session_id}}
        )
        return response, citations


rag_chain = RAGChain()