import argparse
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

def start_sglang_server():
    # 准备参数（模拟命令行参数）
    server_args = ServerArgs(
        model_path="/home/niwang/models/Qwen2.5-1.5B-Instruct",
        host="0.0.0.0",
        port=30000,  # 默认端口
        trust_remote_code=True,
        log_level="info",  # 添加日志级别参数
    )
    
    # 直接调用launch_server函数
    launch_server(server_args)

if __name__ == "__main__":
    start_sglang_server()
    print("Server is running...")