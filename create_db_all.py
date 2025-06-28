from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

loader = TextLoader("./docs/all.txt", encoding="utf-8")
documents = loader.load()


#loader = TextLoader("C:/Users/todai/OneDrive/Desktop/生成AIエンジニア_Lesson23_サンプルアプリ2/customer-contact/docs/all.txt", encoding="utf-8")
#documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
Chroma.from_documents(texts, embeddings, persist_directory="./.db_all")
