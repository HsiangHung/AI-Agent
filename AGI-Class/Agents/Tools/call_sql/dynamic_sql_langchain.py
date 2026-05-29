import pymysql
import json

import os
from openai import OpenAI

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# from dotenv import load_dotenv
# load_dotenv()

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, current_dir])

from models import qwen_model


# # LangChain提供的记忆组件
# from langchain_classic.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

#表的描述
database_schema_string = """
CREATE TABLE IF NOT EXISTS Classes (
    class_id INT PRIMARY KEY COMMENT '班级的ID编号',
    class_name VARCHAR(100) NOT NULL COMMENT '班级的名称'
) ENGINE=InnoDB COMMENT = '班级表';

CREATE TABLE IF NOT EXISTS Students (
    student_id INT PRIMARY KEY COMMENT '学生的唯一性ID编号',
    name VARCHAR(100) NOT NULL COMMENT '学生姓名',
    class_id INT COMMENT '学生所在班级的ID编号，和班级表中的班级ID编号对应'
) ENGINE=InnoDB COMMENT = '学生表';

CREATE TABLE IF NOT EXISTS Scores (
    score_id INT PRIMARY KEY COMMENT '学生成绩表的唯一性ID编号',
    student_id INT COMMENT '学生个人的ID编号，和学生的唯一性ID编号对应',
    subject VARCHAR(100) NOT NULL COMMENT '考试科目，中文名称标识',
    score FLOAT NOT NULL COMMENT '考试科目的分数'
) ENGINE=InnoDB COMMENT = '学生科目成绩表';
"""


def get_mysql_conn():
    # 程序运行前请运行 db_init.py，并确保数据库和表以及表中数据已存在
    # 端口、用户名、密码、数据库IP地址请根据自己的实际情况进行修改
    return pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='hung123456',
        database='agi_class_test',
        charset='utf8mb4'  # 添加推荐的字符集参数
    )

sql_conn = get_mysql_conn()
cursor = sql_conn.cursor()


#专门负责和数据库进行交互的工具
@tool
def ask_database(query):
    """查询数据库的函数。输出是数据库中表的记录"""
    cursor.execute(query)
    records = cursor.fetchall()
    return records


system_prompt = f"""
你是一个AI助手. 请基于SQL数据库的表格回答用户问题.

SQL应该使用这个数据库架构来编写:
{database_schema_string}
SQL查询查询应以纯文本形式返回，而不是JSON格式。
查询应仅包含MySQL支持的语法.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", "你是一个AI助手. 请基于数据库的表格回答用户问题"),
        ("system", system_prompt),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [ask_database]
agent = create_tool_calling_agent(qwen_model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "查询一班的学生数学成绩是多少？"})
# result = agent_executor.invoke({"input": "所有学生的数学平均成绩是多少？"})
print(result)


"""
Benchmark results:

1. "查询一班的学生数学成绩是多少？":

Invoking: `ask_database` with `{'query': "SELECT s.name, sc.score FROM Students s JOIN Scores sc ON s.student_id = sc.student_id JOIN Classes c ON s.class_id = c.class_id WHERE c.class_name = '一班' AND sc.subject = '数学';"}`

(('张三', 85.5), ('李四', 78.0))一班的学生数学成绩如下：

- 张三：85.5分
- 李四：78分

> Finished chain.
{'input': '查询一班的学生数学成绩是多少？', 'output': '一班的学生数学成绩如下：\n\n- 张三：85.5分\n- 李四：78分'}

-----------------------------------------------------------------
2. "所有学生的数学平均成绩是多少？":

Invoking: `ask_database` with `{'query': "SELECT AVG(score) AS average_math_score FROM Scores WHERE subject = '数学';"}`

((85.16666666666667,),)所有学生的数学平均成绩是 85.17（四舍五入到小数点后两位）。
> Finished chain.
{'input': '所有学生的数学平均成绩是多少？', 'output': '所有学生的数学平均成绩是 85.17（四舍五入到小数点后两位）。'}
"""
