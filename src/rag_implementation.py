import os
import shutil
import argparse
from os.path import join, dirname
from dotenv import load_dotenv

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline



def load_env():
    dotenv_path = join(dirname(__file__), '.env')
    # print("dotenv_path: ", dotenv_path)
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(".env loaded.")
    else:
        print("No .env file found.")


def build_vectorstore(directory_path, file_type, vectore_store_path, embedding_model):
    print(f"Loading documents from {directory_path}")
    loader = DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=TextLoader,
    )
    docs = loader.load()

    print(f"Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)

    if os.path.isdir(vectore_store_path):
        print(f"Removing previous vectorstore at {vectore_store_path}")
        shutil.rmtree(vectore_store_path)

    print(f"Creating vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=vectore_store_path
    )
    return vectorstore


def load_llm(model_id):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ.get("HF_HUB_CACHE"))
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=os.environ.get("HF_HUB_CACHE"))
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
    return HuggingFacePipeline(pipeline=pipe), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for RAG with TinyLlama")
    parser.add_argument("--data_path", type=str, default="data/text", help="Path to the folder with .txt files")
    parser.add_argument("--file_type", type=str, default=".txt", help="Type of file to load (e.g., .txt)")
    parser.add_argument("--vectorstore_path", type=str, default="./sql_chroma_db", help="Path to store Chroma DB")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model ID")

    args = parser.parse_args()

    load_env()

    print("Initializing embedding model...")
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en")

    vectorstore = build_vectorstore(args.data_path, args.file_type, args.vectorstore_path, embedding_model)

    llm, tokenizer = load_llm(args.model_id)

    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    
    print("RAG CLI ready. Ask your questions below (type 'exit' to quit):\n")

    MAX_INPUT_TOKENS = 1024

    while True:
        print('-' * 150)
        query = input(">> ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not query:
            continue
        try:
            # Tokenize and truncate query
            tokenized = tokenizer.encode(query, truncation=True, max_length=MAX_INPUT_TOKENS)
            truncated_query = tokenizer.decode(tokenized, skip_special_tokens=True)

            response = qa_chain.invoke(truncated_query)
            print("🔎 Answer:", response["result"])
        except Exception as e:
            print("⚠️ Error:", str(e))


if __name__ == "__main__":
    main()
