"""In this exercice, we are going to do 4 steps,
1 - load the KB on Textloaders
2 - We will chunk it in the chuncking, it as per chunking strategy via Regressive Character splitter with chunksize and chunk overlap
3 - We will create a vector DB
4- We will make a call to groq llm"""



from dotenv import load_dotenv

import os

load_dotenv()
os.getenv("GROQ_API_KEY")


from langchain_classic.document_loaders import TextLoader

loader = TextLoader("data.txt")
doc = loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = splitter.split_documents(doc)


from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


from langchain_community.vectorstores import FAISS

vector_db = FAISS.from_documents(doc,embeddings)



from groq import Groq

client = Groq()


response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [{"role":"system", "content":"Tell me about India in short"}, {"role": "system", "content":"Tell me about Asia in short"}]
)


"""no 5 - additional one is to define and get the retriever so that the top queries can be fetched from db - by .as_retriever"""

retriever = vector_db.as_retriever(search_kwargs = {"k":3})

retrieved = retriever.invoke("Advantage of AI")

from langchain_groq import ChatGroq

llm = ChatGroq(model= "llama-3.3-70b-versatile")

llm_response = llm.invoke("Why is sky blue")


print(retrieved)




