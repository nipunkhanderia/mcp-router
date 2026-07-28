


from langchain_classic.document_loaders import TextLoader

"""first step is to load the test data in the TextLoader"""

loader = TextLoader("data.txt")

doc = loader.load()


"""2nd step is is to split the document in Recursive character text splitter"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)


split = recursive.split_documents(doc)

from groq import Groq

from dotenv import load_dotenv

load_dotenv()
import os
os.getenv("GROQ_API_KEY")

client = Groq()

response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [{
        "role":"system", "content":"why is sky blue"
    }]
)


"""from 3rd step is to create vector db"""

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS 

vec_db = FAISS.from_documents(split, embeddings)

retriever = vec_db.as_retriever(search_kwargs = {"k":3})

retrieval = retriever.invoke("Disadvantages of AI")

print(retrieval)
