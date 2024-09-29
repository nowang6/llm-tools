from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine


model = "/home/niwang/models/Qwen15-14B-Chat"
openai_api_key = "EMPTY" 
openai_api_base = "http://localhost:23333/v1"  



Settings.llm = OpenAILike(model=model, api_key=openai_api_key, api_base=openai_api_base, request_timeout=1.0)


documents = SimpleDirectoryReader("db").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="/home/niwang/models/bge-small-zh-v1.5")
index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("IRONFLIP的设计灵感是什么？")
print(response)