from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
# Khai báo biến
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_db_from_text():
    raw_text = """
    Ngành ngân hàng ở Việt Nam đóng vai trò quan trọng trong nền kinh tế, giúp luân chuyển vốn và hỗ trợ hoạt động kinh doanh. 
    Hiện nay, hệ thống ngân hàng bao gồm Ngân hàng Nhà nước Việt Nam – cơ quan quản lý chính sách tiền tệ, cùng với các ngân hàng thương mại, ngân hàng chính sách và tổ chức tài chính vi mô. 
    Các ngân hàng thương mại lớn như Vietcombank, BIDV, VietinBank và Techcombank cung cấp đa dạng dịch vụ, từ tài khoản tiết kiệm, cho vay cá nhân đến tín dụng doanh nghiệp. 
    Đặc biệt, xu hướng số hóa đang phát triển mạnh mẽ với sự ra đời của ngân hàng số và ví điện tử như Momo, ZaloPay.  
    Dù có sự tăng trưởng mạnh, ngành ngân hàng vẫn đối mặt với thách thức như nợ xấu, áp lực cạnh tranh từ ngân hàng nước ngoài và yêu cầu tuân thủ các tiêu chuẩn quốc tế. 
    Tuy nhiên, với sự phát triển kinh tế và cải cách liên tục, hệ thống ngân hàng Việt Nam được kỳ vọng sẽ ngày càng hiện đại, minh bạch và hiệu quả hơn trong tương lai.
    """
    
    # Khởi tạo text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n", #khi xuống dòng sẽ chia ra
        chunk_size=1024, #mỗi đoạn text là 50 ký tự
        chunk_overlap=256, #khoảng lặp lại giữa 2 đoạn là 50 ký tự
        length_function=len 
    )
    chunks = text_splitter.split_text(raw_text)
    print(chunks)
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    embeddings = [embedding_model.embed_documents(chunk) for chunk in chunks]
    print("Embeddings:", embeddings)
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_files():
    # Khởi tạo document loader
    loader = DirectoryLoader(path=pdf_data_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[1]
    print(document.page_content)
    print(document.metadata)

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )

    # Tạo FAISS vector store từ các Document đã split
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    print("Successfully created and saved FAISS vector store")
    return db

create_db_from_files()


