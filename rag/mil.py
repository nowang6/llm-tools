from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# load documents
documents = SimpleDirectoryReader(
    input_files=["db/售后产品知识库.docx"]
).load_data()


vector_store = MilvusVectorStore(uri="http://localhost:19530", dim=512,
                                 collection_name="vertu")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding(model_name="/home/niwang/models/bge-small-zh-v1.5")
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    embed_model=embed_model
)