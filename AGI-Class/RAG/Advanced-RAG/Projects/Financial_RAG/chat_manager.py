from langchain_classic.memory import ConversationBufferWindowMemory

# ConversationBufferWindowMemory
# from langchain.memory import ConversationBufferWindowMemory  # 修正
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from config import *

class ChatManager:
    def __init__(self):
        self.memory_k = MEMORY_WINDOW

    def get_history(self, session_id: str):
        """获取Redis对话历史"""
        return RedisChatMessageHistory(
            session_id=session_id,
            url=REDIS_URL,
            ttl=86400 * 7
        )

    def get_memory(self, session_id):
        """窗口记忆，自动截断超长历史"""
        return ConversationBufferWindowMemory(
            chat_memory=self.get_history(session_id),
            k=self.memory_k,
            return_messages=True
        )

    def wrap_chain(self, rag_chain):
        """包装RAG链，支持多轮对话"""
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

chat_manager = ChatManager()