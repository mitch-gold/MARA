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
from transformers import pipeline
from langchain_community.callbacks import get_openai_callback
app = Flask(__name__)

OPENAI_API_KEY =os.environ.get('OPENAI_API_KEY')
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7, api_key=OPENAI_API_KEY)

# Global variable for conversation history
conversation_history = []
conversation_summary = ""


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize conversation history
def summarize_conversation(conversation_history):
    conversation_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in conversation_history])
    summary = summarizer(conversation_text, max_length=150, min_length=30, do_sample=False)
    print(summary[0]['summary_text'])
    return summary[0]['summary_text']

# Function to handle the conversation completion
def get_completion(prompt):
    global conversation_history, conversation_summary

    # Check if we need to summarize the conversation (e.g., after every 5 exchanges)
    if len(conversation_history) >= 5:  # Summarize after 5 exchanges
        conversation_summary = summarize_conversation(conversation_history)
        conversation_history = []  # Reset the history for the next set of exchanges
    
    # Load system message
    with open("data/system_message.txt", "r") as f:
        sysMessageText = f.read()

    systemMessage = sysMessageText + """
    Context: {context}

    Conversation Summary:
    {summary}

    Current Prompt:
    {prompt}

    Answer:"""

    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "summary", "prompt"],
        template=systemMessage,
    )

    # Initialize LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Load documents for context (optional)
    loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    persist_directory = "data/persistent_memory_storage"
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    # Retrieve relevant context from documents
    docs = vector_store.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Query LLM with context and summarized conversation history
    with get_openai_callback() as cb:
        response = llm_chain.invoke({"context": context, "summary": conversation_summary, "prompt": prompt})["text"]

        # Update conversation history with the current exchange
        conversation_history.append((prompt, response))
        print(prompt)
        print(cb)
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