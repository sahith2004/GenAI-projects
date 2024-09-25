from fastapi import FastAPI, Body
# from langchain.llms import LLModel

app = FastAPI()

# Replace "your_llm_model" with your actual LLM model instance
# llm_model = LLModel("your_llm_model")
print('starting server')
@app.post("/chat")
async def chat(message: str = Body(...)):
  """
  Chat endpoint that takes a message and returns a response from the LLM model.
  """
  # Prepare the chat history for the LLM model
  chat_history = [{"role": "user", "content": message}]

  # Call the LLM model with the chat history
  #   response = llm_model.run(chat_history)

  # Extract the response from the LLM model output
  # This might vary depending on the specific model you're using
  #   model_response = response[0]["content"]

  # Return the chat response from the LLM model
  return {"message": message}