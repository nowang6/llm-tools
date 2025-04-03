import requests
import json

def call_llm(prompt, max_tokens=2048, temperature=0.7):
    url = "http://localhost:30000/generate"
    
    # 准备请求数据
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False  # 设置为True则使用流式输出
    }
    
    # 发送请求
    response = requests.post(url, json=data)
    
    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        return result["text"][0]  # 返回生成的文本
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def get_model_info():
    """获取模型信息"""
    url = "http://localhost:30000/get_model_info"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    # 首先获取模型信息
    model_info = get_model_info()
    print("Model Info:", json.dumps(model_info, indent=2, ensure_ascii=False))
    
    # 测试调用模型
    prompt = "你好，请介绍一下你自己"
    print("\nPrompt:", prompt)
    response = call_llm(prompt)
    print("\nResponse:", response) 