import os
from dotenv import load_dotenv

# Modern, non-deprecated imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <--- Fixes the error
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import pickle

load_dotenv()

DATA_DIR = "./data_source"
FAISS_INDEX_PATH = "vector_store/faiss_index"
BM25_INDEX_PATH = "vector_store/bm25_retriever.pkl"

def run_ingestion():
    # 1. Create directories if they don't exist
    os.makedirs("vector_store", exist_ok=True)
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Please place your PDF files into the '{DATA_DIR}' directory first!")
        return

    print("🚀 Starting Data Ingestion Pipeline...")
    
    # 2. Load Documents
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages from PDFs.")

    # 3. Smart Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(docs)
    print(f"✂️ Split documents into {len(final_chunks)} semantic chunks.")

    # 4. Initialize Local Embedding Model (Production-grade open-source)
    print("🧠 Generating Dense Vectors via HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # 5. Build and Save FAISS Vector Index (Dense)
    db = FAISS.from_documents(final_chunks, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print("💾 FAISS Vector Index saved successfully.")

    # 6. Build and Save BM25 Retriever (Sparse)
    print("📝 Building Sparse Keyword Index (BM25)...")
    bm25_retriever = BM25Retriever.from_documents(final_chunks)
    bm25_retriever.k = 4  # Pull top 4 matches
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("💾 BM25 Index saved successfully.")
    print("✨ Ingestion completed! You are ready to run the app.")

if __name__ == "__main__":
    # Create an empty data source folder to help the user start
    os.makedirs(DATA_DIR, exist_ok=True)
    run_ingestion()