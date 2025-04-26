from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
def llm_model():
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-2024-08-06",
        temperature=0.01,
    )
    return llm

def embedding_model():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    return embedding_model