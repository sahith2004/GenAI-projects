from fastapi import FastAPI, Body
# from langchain.llms import LLModel
import langchain
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains.question_answering import load_qa_chain
import boto3
from pinecone import Pinecone as PineconeClient
app = FastAPI()

# Set Pinecone API key and index name environment variables
os.environ["PINECONE_API_KEY"] = "c4355e08-39f3-486d-9ac6-67d0b046b08e"
os.environ["PINECONE_INDEX_NAME"] = "quickstart"

# Initialize Bedrock client for embeddings
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Replace "your_llm_model" with your actual LLM model instance
# llm_model = LLModel("your_llm_model")

# Read documents function
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

doc=read_doc('documents/')
len(doc)


# Chunk data function
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)
len(documents)
# Retrieve answers function
def retrieve_answers(query):
    doc_search = retrieve_query(query)
    print(doc_search)
    response = chain.run(input_documents=doc_search, question=query)
    return response

index = PineconeVectorStore.from_documents(documents, bedrock_embeddings, index_name="quickstart")



# Define retrieve query function
def retrieve_query(query, k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

# Load question answering model
llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock, model_kwargs={"max_gen_len": 256})
chain = load_qa_chain(llm, chain_type="stuff")

print('starting server')
@app.post("/chat")
async def chat(message: str = Body(...)):
  """
  Chat endpoint that takes a message and returns a response from the LLM model.
  """
  # Prepare the chat history for the LLM model
  print(message)
  answer = retrieve_answers(message)
  print("rag answer ,", answer)
  # Call the LLM model with the chat history
  #   response = llm_model.run(c`hat_history)

  # Extract the response from the LLM model output
  # This might vary depending on the specific model you're using
  #   model_response = response[0]["content"]

  # Return the chat response from the LLM model
  return {"message": answer}