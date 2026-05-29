import mysql.connector
from config import *
from loguru import logger

class MySQLDB:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # 知识库表
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge_base
                             (id VARCHAR(64) PRIMARY KEY, name VARCHAR(255), create_time DATETIME)''')
        # 文档表
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                             (id VARCHAR(64) PRIMARY KEY, kb_id VARCHAR(64), filename VARCHAR(255), upload_time DATETIME)''')
        # 对话历史表
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history
                             (id VARCHAR(64) PRIMARY KEY, session_id VARCHAR(64), content TEXT, role VARCHAR(20), create_time DATETIME)''')
        self.conn.commit()

    def insert_kb(self, kb_id, name):
        self.cursor.execute("INSERT INTO knowledge_base VALUES (%s, %s, NOW())", (kb_id, name))
        self.conn.commit()

    def insert_doc(self, doc_id, kb_id, filename):
        self.cursor.execute("INSERT INTO documents VALUES (%s, %s, %s, NOW())", (doc_id, kb_id, filename))
        self.conn.commit()

    def get_all_kb(self):
        self.cursor.execute("SELECT id, name FROM knowledge_base")
        return self.cursor.fetchall()

    def delete_kb(self, kb_id):
        try:
            # 先检查知识库是否存在
            self.cursor.execute("SELECT id FROM knowledge_base WHERE id=%s", (kb_id,))
            if not self.cursor.fetchone():
                logger.warning(f"知识库 {kb_id} 不存在于 MySQL 中")
                return False

            # 删除文档
            self.cursor.execute("DELETE FROM documents WHERE kb_id=%s", (kb_id,))
            doc_deleted = self.cursor.rowcount
            # 删除知识库
            self.cursor.execute("DELETE FROM knowledge_base WHERE id=%s", (kb_id,))
            kb_deleted = self.cursor.rowcount
            self.conn.commit()
            logger.info(f"删除 MySQL 记录：知识库 {kb_deleted} 条，文档 {doc_deleted} 条")
            return kb_deleted > 0
        except Exception as e:
            self.conn.rollback()
            logger.error(f"删除 MySQL 失败：{e}")
            return False

mysql = MySQLDB()