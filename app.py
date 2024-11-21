#Final app.py 
#import files
from flask import Flask, render_template, request
import openai
import os 
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
app = Flask(__name__)

OPENAI_API_KEY="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"

llm = OpenAI(model='gpt-3.5-turbo-instruct',
            temperature=0.7,
            api_key=OPENAI_API_KEY)


conversation_history = []


def get_completion(prompt):
# Prompt template with conversation history
    with open("data/system_message.txt","r") as f:
        sysMessageText = f.read()

    systemMessage = sysMessageText + """
    Context: {context}

    Conversation History:
    {history}

    Current Prompt:
    {prompt}

    Answer:"""

    prompt_template = PromptTemplate(
        input_variables=["context", "history", "prompt"],
        template=systemMessage,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    persist_directory = "data/persistent_memory_storage"
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    # Retrieve relevant context from documents
    docs = vector_store.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format conversation history
    history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in conversation_history])

    # Query LLM with context and history
    response = llm_chain.invoke({"context": context, "history": history, "prompt": prompt})["text"]

    # Update conversation history
    conversation_history.append((prompt, response))

    return response
@app.route("/")
def home():    
    return render_template("index.html")
@app.route("/get")

def get_bot_response():    
    userText = request.args.get('msg')  
    response = get_completion(userText)  
    #return str(bot.get_response(userText)) 
    return response
if __name__ == "__main__":
    app.run()