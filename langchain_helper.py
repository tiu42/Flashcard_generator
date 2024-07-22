from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

embeddings = OpenAIEmbeddings()

class Flashcard(BaseModel):
    """Flashcard generated from document"""
    front: str = Field(description = "The content on the front of the card")
    back: str = Field(description = "The content on the back of the card")

class Deck(BaseModel):
    """List of all flashcards generated"""
    deck: List[Flashcard]

def create_vector_db(pdf_path)->FAISS:
    loader = PyPDFLoader(file_path=pdf_path, extract_images=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_response(query, db, k=5):
    docs = db.similarity_search(query, k=k)
    text = " ".join([d.page_content for d in docs])
    llm = ChatOpenAI(model="gpt-4o")
    structured_llm = llm.with_structured_output(Deck.schema())
    message = [
        ("system", "You are a professional Anki card creator, able to create Anki cards from the text provided. "
         "Regarding the formulation of the card content, you stick to two principles: "
         "First, minimum information principle: The material you learn must be formulated in as simple way as it is only possible. Simplicity does not have to imply losing information and skipping the difficult part. "
         "Second, optimize wording: The wording of your items must be optimized to make sure that in minimum time the right bulb in your brain lights up. This will reduce error rates, increase specificity, reduce response time, and help your concentration. "),
        ("human", "Make 20 flashcards about this topic: {question} by searching the following document: {docs}.")
    ]
    prompt_template = ChatPromptTemplate.from_messages(message)
    chain = prompt_template | structured_llm
    response = chain.invoke({'question': query, 'docs': text})
    return response