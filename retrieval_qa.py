import os

from langchain_community.document_loaders import PythonLoader, TextLoader, GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from tempfile import TemporaryDirectory

load_dotenv()

documents = []

repo_url = os.environ.get("REPO_URL")

with TemporaryDirectory() as tempdir:
    loader = GitLoader(repo_path=tempdir + r"/repo", clone_url=repo_url, branch="main")
    documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_texts([doc.page_content for doc in chunks], OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        retriever=retriever,
        chain_type="stuff"
    )

    response = qa_chain.run("""
    Give me an overview of the repository
    """)
    print(response)
