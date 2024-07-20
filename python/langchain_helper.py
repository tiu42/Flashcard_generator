from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db(pdf_path)->FAISS:
    loader = PyPDFLoader(file_path=pdf_path, extract_images=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_response(db, k=4):
    query = "Summarize this document"
    docs = db.similarity_search(query, k=k)
    text = " ".join([d.page_content for d in docs])
    print(text)
    print("\n")
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables= ["docs"],
        template= """You are a professional Anki card creator, able to create Anki cards from the text I provide.
        Regarding the formulation of the card content, you stick to two principles:
        First, minimum information principle: The material you learn must be formulated in as simple way as it is only possible. Simplicity does not have to imply losing information and skipping the difficult part.
        Second, optimize wording: The wording of your items must be optimized to make sure that in minimum time the right bulb in your brain lights up. This will reduce error rates, increase specificity, reduce response time, and help your concentration.
        Please create 20 cards from the following text: {docs}
        Output each card you created as a Python dictionary.
        """
    )
    chain = LLMChain(llm =llm, prompt = prompt, output_key = "ans")
    response = chain.invoke({'question': query, 'docs': text})
    response['ans'] = response['ans'].replace("\n"," ")
    return response['ans']

db = create_vector_db('Harry_Potter_1.pdf')
response = get_response(db)
print(response)