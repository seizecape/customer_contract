import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# ドキュメントの読み込み
loader = TextLoader("./docs/sample.txt", encoding="utf-8")
documents = loader.load()

# テキストをチャンクに分割
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# ベクトルDBの作成
embeddings = OpenAIEmbeddings()
Chroma.from_documents(texts, embeddings, persist_directory="./.db_customer")

print("✅ .db_customer データベース作成完了")
