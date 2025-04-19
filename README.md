# 🛡️ Use RAG to Enhance Retrieve With LLM

Phân tích lỗ hổng bảo mật trong Smart Contracts thông qua Chatbot AI ứng dụng RAG (Retrieval-Augmented Generation).

---

## 🚀 Mục tiêu

Dự án sử dụng các mô hình ngôn ngữ lớn (LLM) kết hợp với cơ chế truy xuất tri thức (RAG) để:

- Hiểu và phân tích hợp đồng thông minh
- Tìm kiếm và giải thích các lỗ hổng bảo mật
- Giao tiếp với người dùng thông qua chatbot AI thân thiện

---

## 🧠 Công nghệ sử dụng

- **Python**
- **LangChain**
- **Qwen2-1.5B-Instruct** (qua Hugging Face)
- **OpenAI GPT-4o mini** (tùy chọn)
- **FAISS Vector Database**
- **Hugging Face Transformers**

---

## 🏗️ Cấu trúc thư mục

```
VULNHUNT_GPT/
├── data/                 # Chứa dữ liệu đầu vào hoặc smart contracts
├── miai/                 # (Tuỳ chỉnh: có thể chứa AI config hoặc xử lý riêng)
├── other/                # Mục phụ khác
├── rag-venv/             # Virtual environment (nên được gitignore)
├── vectorstores/         # Lưu FAISS vector db
├── .env                  # File chứa các biến môi trường (API Keys)
├── .gitignore
├── llm_model.py          # Định nghĩa mô hình & pipeline LangChain
├── qabot-gpt.py          # Khởi chạy chatbot sử dụng OpenAI GPT
├── qabot-qwen.py         # Khởi chạy chatbot sử dụng Qwen2-1.5B
├── upload_data.py        # Tải và nhúng dữ liệu vào vector store
└── README.md
```

---

## ⚙️ Cài đặt

### 1. Clone repo:

```bash
git clone https://github.com/2uaan1ee/basic-rag-llm.git
cd vulnhunt-gpt-llm-rag
```

### 2. Cài đặt môi trường:

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API keys:

Tạo file `.env` với nội dung:

```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

> Đảm bảo bạn đã đăng nhập bằng `huggingface-cli login` nếu sử dụng HuggingFace.

---

## 📦 Chuẩn bị dữ liệu

```bash
python upload_data.py
```

Dữ liệu sẽ được xử lý và lưu dưới dạng vector FAISS trong thư mục `vectorstores`.

---

## 🧪 Chạy chương trình

### Với GPT-4o (qua OpenAI):

```bash
python qabot-gpt.py
```

### Với Qwen2-1.5B (qua Hugging Face):

```bash
python qabot-qwen.py
```

---

## 📌 Ghi chú

- Dự án hỗ trợ cả mô hình cloud (OpenAI) và local (Qwen2).
- Đảm bảo RAM tối thiểu 8GB nếu sử dụng Qwen local.
- Tùy chọn GPU để tăng tốc khi dùng Hugging Face models.

---

## 📜 License

MIT License

---

## 🙌 Đóng góp

Mọi đóng góp, chỉnh sửa hoặc mở rộng dự án đều rất hoan nghênh! Bạn có thể mở issue hoặc tạo pull request.

---

## 📬 Liên hệ

Nếu bạn có thắc mắc, hãy liên hệ qua GitHub Issues hoặc email (tuỳ chọn nếu muốn thêm).
