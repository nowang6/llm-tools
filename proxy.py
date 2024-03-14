import os

# 设置代理地址和端口号
proxy_address = "127.0.0.1"
proxy_port = "7890"

# 设置HTTP代理
os.environ["HTTP_PROXY"] = f"http://{proxy_address}:{proxy_port}"
os.environ["HTTPS_PROXY"] = f"http://{proxy_address}:{proxy_port}"

# 可选：设置FTP代理
os.environ["FTP_PROXY"] = f"ftp://{proxy_address}:{proxy_port}"

# 可选：设置SOCKS代理
os.environ["SOCKS_PROXY"] = f"socks://{proxy_address}:{proxy_port}"