
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

import os
load_dotenv()


def load_raw_documents(path:str):
    print(path)
    directoryLoader= DirectoryLoader(path=path, glob="**/*.txt", loader_cls= TextLoader)
    raw_doc_list= directoryLoader.load()
    return raw_doc_list

def create_store_vector_db() :
    raw_documents= load_raw_documents("data")
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        add_start_index= True)
    chunks = text_splitter.split_documents(documents=raw_documents)
    google_gen_ai_embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = FAISS.from_documents(chunks,google_gen_ai_embeddings)
    db.save_local("faiss_index_local")



def get_simlar_documents_from_vectorstore(query:str):
    google_gen_ai_embedding_reader= GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")    
    local_faiss_db = FAISS.load_local("faiss_index_local",google_gen_ai_embedding_reader,allow_dangerous_deserialization= True)
    simialar_documents = local_faiss_db.similarity_search("how to prune orange trees")
    return simialar_documents



print("Ask ypur query please : \n")
query_text = input()
simialar_documents= get_simlar_documents_from_vectorstore(query_text)

context_text = "\n\n".join([chunk.page_content for chunk in simialar_documents])

promt_text = """
answer the question based on following context 
{context} 
---
answer the question based on above context : {query}"""

chat_template = ChatPromptTemplate.from_messages(context_text)
chat_template.format(context= context_text, query= "query text")

myconfig: RunnableConfig = {
    "tags": ["my-custom-tag"],
    "metadata": {"user_id": "123"},
    "max_retries": 3, # Example of a common config option
    "configurable": {
        # Example of configuring a specific field if the model/chain supports it
        # "model_temperature": 0.7
    }
}

#os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"]= os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chain = chat_template | model | StrOutputParser()
response = chain.invoke({"query":query_text, "context":"context_text"})
print(response)
###

