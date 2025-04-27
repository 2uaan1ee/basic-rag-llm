from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from llm_model import get_hf_llm

# Load environment variables
load_dotenv()
VECTOR_DB_PATH = "vectorstores/db_faiss"
DATA_DIR = "data"

def load_documents(data_dir):
    documents = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            documents.extend(TextLoader(file_path).load())
    return documents

def split_documents(documents, chunk_size=1024, chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    for idx, text in enumerate(texts):
        text.metadata["id"] = idx
    return texts

def read_vector_db(vector_db_path):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db

def main():
    # Input query
    print("Your input here:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    query = "\n".join(lines)
    # Initialize LLM and embedding
    llm = get_hf_llm(temperature=0.01)

    # Load FAISS vector store
    db = read_vector_db(VECTOR_DB_PATH)

    # Load and split documents (for BM25)
    documents = load_documents(DATA_DIR)
    texts = split_documents(documents)


    # FAISS retriever
    faiss_retriever = db.similarity_search(query, k=30)
    bm25_retriever = BM25Retriever.from_documents(faiss_retriever, k=10)

    # Prompt template
    template = '''
    Known information: 
    {context}

    Based on the above known information in your vector database, respond to the user's question concisely and professionally. 
    If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided'. 
    Please respond in English.

    The question is: {question}

    Answer:
    '''
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # QA chain
    qa = RetrievalQA.from_llm(
        llm=llm,
        retriever=bm25_retriever,
        prompt=prompt,
        return_source_documents=True
    )
    result = qa.invoke({"query": query})
    print("\nAnswer:\n", result["result"])

if __name__ == "__main__":
    main()