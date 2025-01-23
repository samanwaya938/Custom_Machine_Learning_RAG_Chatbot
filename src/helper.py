from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


def file_loader(path):
  loader = DirectoryLoader(
    path, glob="*.pdf", loader_cls=PyPDFLoader
  )
  documents = loader.load()
  return documents


def chunking_data(data):
  split_data = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap = 50)
  chunk_data = split_data.split_documents(data)
  return chunk_data


def get_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


