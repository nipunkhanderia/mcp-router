"""We are creating a rag pipepline from 
a context of text file outlining advantage of AI - This is the docstring for this file
Thjere will be one full file and then there will be separate methods for the same"""


"""First we need to create test data in txt file so then it can be loade din to this rag pipeline
AI test data created as data.txt"""

"""Now as first step is we have to load the test document inside the langchain ecosystem, so the docunments
needs to loaded into a textloader"""
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from groq import Groq
load_dotenv()
os.getenv("GROQ_API_KEY")

text_loader = TextLoader("data.txt")
txt = text_loader.load()

client = Groq()

response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [
        {"role": "system", "content": "Who is Nipun Khanderia"}
    ]
)

print (response)


# print(text_loader.load())
print(os.getenv("GROQ_API_KEY"))





"""After loading the data in TextLoader we need to come up with chuncking strategy, we need
to cut the input text in to chunks which makes sense and mwhich can be passed on to llm as context for it to give better answers
so we need to get a character spolitter, in this case, recursive character splitter"""

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
splits = splitter.split_documents(txt)



print (splits)












