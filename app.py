#Final app.py 
#import files
from flask import Flask, render_template, request
import os 
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from transformers import pipeline
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

app = Flask(__name__)

OPENAI_API_KEY =os.environ.get('OPENAI_API_KEY')
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7, api_key=OPENAI_API_KEY)

# Token counter
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Truncate text to fit within the token limit
def truncate_text(text, max_tokens, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

def preprocess_documents(directory):
    loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    return split_docs

# Global variable for conversation history
conversation_history = []
conversation_summary = ""


summarizer = pipeline("summarization", model="t5-small")

# Function to summarize conversation history
def summarize_conversation(conversation_history):
    recent_conversations = conversation_history[-3:]  # Last 3 exchanges
    conversation_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent_conversations])
    summary = summarizer(conversation_text, max_length=75, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Function to handle the conversation completion
def get_completion(prompt):
    global conversation_history, conversation_summary

    # Check if we need to summarize the conversation
    if len(conversation_history) >= 3:
        conversation_summary = summarize_conversation(conversation_history)
        conversation_history = []

    # Prepare the system message
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

    # Load documents for context
    documents = preprocess_documents("data/")

    persist_directory = "data/persistent_memory_storage"
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    # Retrieve relevant context from documents
    docs = vector_store.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Count and truncate tokens
    context_tokens = count_tokens(context)
    summary_tokens = count_tokens(conversation_summary)
    prompt_tokens = count_tokens(prompt)

    MAX_TOKENS = 4096
    RESPONSE_TOKENS = 500
    INPUT_TOKENS_LIMIT = MAX_TOKENS - RESPONSE_TOKENS

    if context_tokens + summary_tokens + prompt_tokens > INPUT_TOKENS_LIMIT:
        context = truncate_text(context, max_tokens=INPUT_TOKENS_LIMIT - (summary_tokens + prompt_tokens))

    # Query the LLM
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    with get_openai_callback() as cb:
        response = llm_chain.invoke({
            "context": context,
            "summary": conversation_summary,
            "prompt": prompt
        })["text"]

        conversation_history.append((prompt, response))
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