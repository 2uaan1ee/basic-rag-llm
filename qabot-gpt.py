from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
load_dotenv()
# Khai báo biến
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
vector_db_path = "vectorstores/db_faiss"
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# Define paths
vector_db_path = "vectorstores/db_faiss"

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4o-2024-08-06",
    temperature=0.01,
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

# Main logic
if __name__ == "__main__":
    # Load the FAISS vector store
    db = read_vector_db()

    # Create a retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

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
    # Define the prompt template
    template = '''
    Known information: 
    {context}
    Based on the above known information in your vector database, respond to the user's question concisely and professionally. 
    If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,'. 
    Please respond in English. The question is 
    {question}
    Answer:
    '''
    prompt = create_prompt(template)

    # Create the QA chain
    qa_chain = create_qa_chain(prompt, llm, retriever)

    # Example query
    result = qa_chain({"query": query})

    # Display the answer and sources
    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content}")
        print("-----------")