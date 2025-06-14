# B Parent Query Assistant (RAG Chatbot)

This project is a sophisticated AI assistant built for Assignment 3. It's designed to help parents get instant answers to frequently asked questions about educational programs, pricing, and partnerships. The core of this application is a **Retrieval-Augmented Generation (RAG)** pipeline, which allows the AI to answer questions based on a specific, private knowledge base rather than its general pre-trained knowledge.

The application features a user-friendly chat interface built with Streamlit and is powered by Google's Gemini model through the LangChain framework.


*(Feel free to replace this with a screenshot of your own running application!)*

---

## ✨ Key Features

-   **Retrieval-Augmented Generation (RAG):** The chatbot doesn't use its general knowledge. Instead, it "retrieves" relevant information from a provided text document to "generate" a contextually accurate answer.
-   **Localized Knowledge Base:** The information source has been tailored to an Indian context, featuring relevant exam names (IIT-JEE, NEET), pricing in Rupees (₹), and local school partnership models.
-   **Conversational Interface:** A clean, chat-like UI built with Streamlit allows for natural interaction.
-   **Built-in Feedback System:** Users can rate the helpfulness of each answer (👍 / 👎), providing a mechanism for future model improvement.
-   **Modular and Modern Code:** Uses the latest LangChain Expression Language (LCEL) for building chains, ensuring the code is up-to-date with best practices.

---

## 🛠️ Technology Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **LLM Framework:** [LangChain](https://www.langchain.com/)
-   **LLM:** [Google Gemini](https://deepmind.google/technologies/gemini/) (via `langchain-google-genai`)
-   **Vector Store:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) - For efficient in-memory similarity searches.
-   **Embeddings:** Google Generative AI Embeddings (`models/embedding-001`)

---

## 🚀 How to Run Locally

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

-   Python 3.9+
-   Git

### 2. Clone the Repository

```bash
git clone https://github.com/[Your-GitHub-Username]/[your-repo-name].git
cd [your-repo-name]
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

The application needs a Google API key to work.

-   Create a new file named `.env` in the root directory of the project.
-   Open the `.env` file and add your API key in the following format:

    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

### 6. Run the Streamlit App

You're all set! Run the following command in your terminal:

```bash
streamlit run app.py
```

The application should now be open and running in your web browser.

---

## 🧠 How the RAG Pipeline Works

This project's core logic follows a simple but powerful pattern:

1.  **Load & Chunk:** The static `knowledge_base_text` is loaded and split into smaller, manageable chunks.
2.  **Embed & Store:** Each chunk of text is converted into a numerical vector (an "embedding") using Google's embedding model. These vectors are then stored in a FAISS vector store for fast retrieval.
3.  **Retrieve:** When a user asks a question, their question is also converted into an embedding. The FAISS store is then queried to find the text chunks with the most similar embeddings (i.e., the most relevant information).
4.  **Generate:** The retrieved chunks of text are combined with the original question into a detailed prompt. This prompt is then sent to the Gemini LLM, which generates a final, human-readable answer based *only* on the context provided.
