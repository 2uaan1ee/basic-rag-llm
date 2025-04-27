from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
VECTOR_DB_PATH = "vectorstores/db_faiss"

def llm():
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="gpt-4o-2024-08-06",
        temperature=0.01,
    )
    return llm

def embedding_model():
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embedding_model

def read_vector_db():
    db = FAISS.load_local(
        folder_path=VECTOR_DB_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db

def load_and_split_documents(data_dir):
    documents = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            documents.extend(TextLoader(file_path).load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    for idx, text in enumerate(texts):
        text.metadata["id"] = idx
    return texts

def response_llm():
    print("Your input here:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    query = "\n".join(lines)

    template = '''
    Known information: 
    {context}
    You are an AI Auditor about the Smart Contract Vulnerabilities. Based on the above known information in your vector database, respond to the user's question concisely and professionally. 
    If the smart contract is not has vulnerabilities, please answer "It works, no vulnerabilities found". 
    Please respond in English.
    The question is: {question}
    Answer:
    '''

    db = read_vector_db()

    # Compression retriever
    faiss_retriever = db.similarity_search(query=query, k=30)
    bm25_retriever = BM25Retriever.from_documents(faiss_retriever, k=10)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    qa = RetrievalQA.from_llm(
        llm=llm(),
        retriever=bm25_retriever,
        prompt=prompt,
        return_source_documents=False
    )
    result = qa.invoke({"query": query})
    print("\nAnswer:\n", result["result"])

if __name__ == "__main__":
    response_llm()