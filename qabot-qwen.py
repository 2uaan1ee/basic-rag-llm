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
from app.llms.llm_model import get_hf_llm
# Load environment variables
load_dotenv()
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

api_key_openai = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = get_hf_llm(temperature=0.01)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Function to create a prompt
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Function to create a QA chain
def create_qa_chain(prompt, llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # Ensure source documents are returned
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

# Function to read the FAISS vector database
def read_vector_db():
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,  # Add this parameter for compatibility
    )
    return db

# Function to format retrieved documents
def format_docs(docs):
    formatted = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )
    return formatted


if __name__ == "__main__":
    # Load the FAISS vector store
    db = read_vector_db()
    
    # Load all .txt files from the 'data' directory
    data_dir = "data"
    documents = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            documents.extend(TextLoader(file_path).load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    texts = text_splitter.split_documents(documents)

    # Add metadata to each chunk
    for idx, text in enumerate(texts):
        text.metadata["id"] = idx

    # Create embeddings and retriever
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    retriever = FAISS.from_documents(texts, embedding).as_retriever(
        search_kwargs={"k": 30}
    )
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 10  # Retrieve top 10 results
    query = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.20;

    contract HelloWorld {

        string public greeting;

        // Constructor to set the initial greeting message
        constructor() {
            greeting = "Hello, World!";
        }

        // Function to get the greeting message
        function getGreeting() public view returns (string memory) {
            return greeting;
        }

        // Function to change the greeting message
        function setGreeting(string memory newGreeting) public {
            greeting = newGreeting;
        }
    }


    What is the vulnerabilities of this code? Give me solutions.
    """

    # Define the prompt template
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
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], 
        weights=[0.2, 0.8]
    )
    
    # Create the cohere rag retriever using the chat model
    docs = ensemble_retriever.invoke(query)
    qa = RetrievalQA.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        prompt=prompt,
        return_source_documents=True
    )
    result = qa.invoke({"query": query})
    print("\nAnswer:\n", result["result"])