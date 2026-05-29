#
#  This code books tickets via 12306 website. Using tools:
#  1. Web crawler
#
#  This code corresponds to lecture L3.20
#
import json
# import requests
import inspect
# from models import get_normal_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_PLUS_MODEL
import pandas as pd
# from datetime import datetime

# client = get_normal_client()

from tools import (
    check_tick,
    check_date,
)

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, current_dir])
from models import client


ALI_TONGYI_PLUS_MODEL = "qwen-plus-2025-09-11"
# 定义函数映射字典
function_map = {
    "check_tick": check_tick,
    "check_date": check_date,
}


def get_completion(messages, model=ALI_TONGYI_PLUS_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_tick",
                    "description": "给定日期查询有没有票",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "日期",
                            },
                            "start": {
                                "type": "string",
                                "description": "出发站的地址编码",
                            },
                            "end": {
                                "type": "string",
                                "description": "终点站的地址编码",
                            }

                        },

                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_date",
                    "description": "返回当前的日期",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "str": {
                                "type": "string",
                                "description": "返回今天的日期",
                            }
                        }
                    }
                }
            }
        ]
    )
    return response.choices[0].message


if __name__ == "__main__":

    prompt = "查询今天北京到上海的票"

    messages = [
        {"role": "system", "content": "你是一个地图通，你可以找到任何地址，找到地址后可以参考的地址编码有<北京：BJP；上海：SHH；天津：TJP；长沙：CSQ；>"},
        {"role": "user", "content": prompt}
    ]
    response = get_completion(messages)

    messages.append(response)  # 把大模型的回复加入到对话中
    print("=====大模型回复=====")
    print(response)
    print()

    # 用户的请求需要多次函数调用，如果返回的是函数调用结果，则打印出来
    while (response.tool_calls is not None):
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print("参数：", args)

            function_name = tool_call.function.name
            if function_name in function_map:
                print(f"Call: {function_name}")
                func = function_map[function_name]
                # 获取函数签名，python内置内省库inspect
                sig = inspect.signature(func)
                params = sig.parameters

                # 根据函数参数决定如何调用
                if params:  # 函数有参数
                    if args:
                        result = func(**args)
                    else:
                        # 可以提供默认值或抛出错误
                        result = func()
                else:  # 函数无参数
                    result = func()

            print(f"=====函数{function_name}返回=====")
            print(result)

            messages.append({
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)  # 数值result 必须转成字符串
            })

        response = get_completion(messages)
        print("=====大模型回复2=====")
        print(response)
        messages.append(response)  # 把大模型的回复加入到对话中
        print()

    print("=====最终回复=====")
    print(response.content)