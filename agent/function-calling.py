import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Qwen2
base_url = "http://192.168.0.250:18080/v1"
api_key = "EMPTY"
model = "qwen2-instruct"


# # glm4
# base_url = "http://192.168.0.250:18080/v1"
# api_key = "EMPTY"
# model = "glm4-chat"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": "用一个关键字搜索图片",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "用于搜索图片的关键字。 比如：南天落地方案",
                    }
                },
                "required": ["keyword"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_generation",
            "description": "使用一个指令生成一张图片",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "用于生成图片的指令, 比如：请生成一只可爱的猫",
                    }
                },
                "required": ["prompt"]
            },
        }
    },
     {
        "type": "function",
        "function": {
            "name": "descripe_image",
            "description": "描述一张图片",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_uri": {
                        "type": "string",
                        "description": "这张图片的URI地址，比如： https://example.com/image.jpg",
                    },
                    "question": {
                        "type": "string",
                        "description": "针对图片的提问，比如： 请描述图片",
                    },
                },
                "required": ["image_uri","question"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "other_functions",
            "description": "不属于以上的函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "用户的原始指令或者问题，比如： 南天是什么时候成立的？",
                    }
                },
                "required": ["prompt"]
            },
        }
    },
]


questions = [
    "请帮我搜索一张蓝天落地方案的图片",
    "帮我搜索一张孙杨的图片",
    "帮我找一张狗的图片",
    "帮我生成一张可爱的猫的照片",
    "请描述一下这张照片https://example.com/image.jpg",
    "这张照片里的技术结果是什么https://example.com/image.jpg",
    #"南天是什么时候成立的",
    #"你是谁训练的"
]


for question in questions:
    messages = []
    messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
    messages.append({"role": "user", "content": question})
    chat_response = chat_completion_request(
        messages, tools=tools
    )
    assistant_message = chat_response.choices[0].message
    print("Question: ", question)
    print("Function name:", assistant_message.tool_calls[0].function.name)
    print("Function argument:", assistant_message.tool_calls[0].function.arguments)
    
