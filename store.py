from src.helper import *
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as pinecone
from dotenv import load_dotenv
import os


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


file = file_loader()
chunk_data = chunking_data(file)
embeddings = get_embedding()




pc = pinecone(PINECONE_API_KEY)

index_name = "mlragchatbot"

pc.create_index(name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))


store_data = PineconeVectorStore.from_documents(
    documents=chunk_data, 
    embedding=embeddings, 
    index_name=index_name
)