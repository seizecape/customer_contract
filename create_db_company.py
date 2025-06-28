from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

loader = TextLoader("./docs/company.txt", encoding="utf-8")  # ← encoding を追加

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
Chroma.from_documents(texts, embeddings, persist_directory="./.db_company")
