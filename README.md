# RAG Context Engine 🚀

Welcome to the **RAG Context Engine**! If you are new to AI or programming, don't worry—this project is designed to be highly modular and easy to understand. 

This system is a "Retrieval-Augmented Generation" (RAG) pipeline. In plain English, it's an AI assistant that reads *your* documents (like PDFs or text files) to answer questions, rather than just guessing or making things up.

---

## 📖 Beginner Guides
If you want to understand how this code works without getting lost in technical jargon, start by reading these guides (located in the documentation artifacts):
1. **[Terminology Guide]**: Translates heavy AI jargon into simple concepts.
2. **[File-by-File Guide]**: Explains what every Python file does and why it exists.
3. **[Code Workflow]**: Traces exactly what happens behind the scenes when a user asks a question.

---

## 🛠️ Setup Instructions

Follow these steps to get the system running on your computer.

### 1. Install Dependencies
You need Python installed on your computer. Open your terminal in this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Set Up Your Environment Variables
The system needs API keys to talk to the AI (Groq) and the database (Weaviate). 
Create a file named `.env` in this folder and add the following lines, replacing the placeholders with your actual keys:

```text
GROQ_API_KEY="your_groq_api_key_here"
WEAVIATE_URL="your_weaviate_cluster_url_here"
WEAVIATE_API_KEY="your_weaviate_api_key_here"
```

### 3. Run the Server
Once everything is installed and your keys are set, start the server by running:
```bash
python main.py
```

### 4. Open the Web Interface
When the server starts successfully, open your web browser and go to:
**http://localhost:8000**

You will see the user interface where you can upload documents and start asking questions!

---

## 🧩 Project Structure
Here is a quick map of where everything lives:

*   **`main.py`**: The front door. It runs the web server and hosts the user interface.
*   **`core/`**: The brain of the application. Contains all the specialized modules (like the `retriever` for searching and the `compressor` for cleaning up text).
*   **`index.html` & `static/`**: The frontend user interface (buttons, chat windows, styling).
*   **`memory.db`**: A local database file that is automatically created to save your chat history.
