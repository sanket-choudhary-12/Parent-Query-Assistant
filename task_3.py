import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


st.set_page_config(
    page_title="Parent Query Assistant",
    page_icon=" B",
    layout="wide"
)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API Key not found. Please create a .env file and add your key.")
    st.stop()


knowledge_base_text = """
### Premier Coaching Programs - Overview ###

Our flagship offering is the **Premier Engineering & Medical Entrance Program (PEMP)**. This is a comprehensive 2-year integrated program for students in Class 11 and 12, targeting the nation's most competitive entrance exams: IIT-JEE (Main & Advanced) for Engineering and NEET for Medical colleges. The PEMP curriculum includes intensive classroom coaching, rigorous mock test series, doubt-clearing sessions, and all-inclusive study materials designed by our expert faculty.

Our **Commerce & Law Achievers Program (CLAP)** is a specialized track for students aiming for top-tier careers in Commerce and Law. This program prepares students for the Common University Entrance Test (CUET) for leading universities like Delhi University, as well as professional entrance exams like CLAT (for National Law Universities) and the CA Foundation (for Chartered Accountancy).

### Fees and Scholarship Structure ###

- **Premier Engineering & Medical Program (PEMP):** The total fee is ₹2,50,000 for the full two-year course. We offer a convenient installment plan: ₹1,00,000 at the time of admission, followed by four quarterly installments of ₹37,500. A 10% discount is available for a one-time full payment.

- **Commerce & Law Achievers Program (CLAP):** The fee is ₹1,80,000 for the two-year program. The installment plan is ₹60,000 at admission and four quarterly installments of ₹30,000.

- **Merit-Based Scholarships:** We offer scholarships of up to 50% on tuition fees for students who have scored above 95% in their Class 10 Board Exams (CBSE/ICSE).

- **One-on-One Mentoring Session:** For targeted guidance, we offer one-on-one sessions with our senior faculty at a rate of ₹2,500 per hour. This is ideal for specific doubt resolution or strategy planning.

- **Crash Courses & Workshops:** We conduct intensive 15-day crash courses before major exams, priced at ₹25,000. We also offer weekend workshops on topics like "Advanced Problem-Solving for JEE" priced at ₹5,000.

### School Alliances ###

We have a prestigious academic alliance with **The Modern India School, Vasant Vihar**, where our PEMP is offered as an integrated school program. Our faculty conducts classes directly on their campus, streamlining the preparation process for their students.

Furthermore, we collaborate with the **'Akanksha Education Trust'**, a non-profit organization dedicated to educational upliftment. Through this partnership, we provide 20 fully-funded seats in our PEMP program to meritorious students from economically weaker sections. The selection for these seats is managed exclusively by the Trust. At present, we do not have official tie-ups with government or state board schools.
"""
@st.cache_resource
def setup_rag_pipeline(text_data, api_key):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=text_data)]
    split_docs = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)

    # This is the new, modern way to structure the prompt and chain
    prompt_template_str = """
    You are an AI assistant for parents interested in our counseling programs.
    Use the following context to answer the question. Your tone should be helpful, professional, and clear.
    If you don't know the answer based on the provided context, just say, "I'm sorry, I don't have information on that topic. Please contact our support team for more details."
    Do not make up information.

    CONTEXT:
    {context}

    QUESTION:
    {input}

    ANSWER:
    """

    prompt = PromptTemplate.from_template(prompt_template_str)

    # This chain takes the question and the retrieved documents and generates an answer.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # This is the primary chain that ties the retriever and document_chain together.
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Main Application UI ---

st.title("B Parent Query Assistant")
st.markdown("Welcome! Ask me anything about our programs, pricing, or school partnerships.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Feedback Collection (Assignment Task 3) ---

def handle_feedback(feedback_type):
    st.session_state.feedback_given = True
    st.toast(f"Thank you for your feedback! We've recorded that the answer was {feedback_type}.")

if not st.session_state.feedback_given:
    st.markdown("---")
    cols = st.columns(2)
    with cols[0]:
        st.button("👍 Helpful", on_click=handle_feedback, args=("helpful",), use_container_width=True)
    with cols[1]:
        st.button("👎 Not Helpful", on_click=handle_feedback, args=("not helpful",), use_container_width=True)

# --- Chat Input and RAG Invocation ---

if user_question := st.chat_input("What is your question?"):

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Finding the best answer for you..."):
        try:
            rag_chain = setup_rag_pipeline(knowledge_base_text, google_api_key)
            
            # The new way to invoke the chain
            response = rag_chain.invoke({"input": user_question})
            
            # The answer is now in the 'answer' key
            ai_answer = response['answer']

            with st.chat_message("assistant"):
                st.markdown(ai_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            st.session_state.last_question = user_question
            st.session_state.last_answer = ai_answer
            st.session_state.feedback_given = False
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")