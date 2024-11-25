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
from langchain.docstore.document import Document
import tiktoken
import json
from funcs.doc_processing import load_json_to_documents, create_vector_store
from funcs.limit_tokens import count_tokens, truncate_text

### IGNORING THESE FOR NOW ###
import warnings
# Suppress LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
### ###


app = Flask(__name__)

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.7, api_key=OPENAI_API_KEY)

#TOKEN SET UP
MAX_TOKENS = 4096
RESPONSE_TOKENS = 500
INPUT_TOKENS_LIMIT = MAX_TOKENS - RESPONSE_TOKENS

# File paths
json_file = "data/mitchell.json"
persist_directory = "data/persistent_memory_storage"

# Load documents and create vector store
documents = load_json_to_documents(json_file)  # Load documents from JSON file
vector_store = create_vector_store(documents, persist_directory)  # Create Chroma vector store


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

    conversation_summary = summarize_conversation(conversation_history) #conversation_summary will always only use the last 3 exchances
    if len(conversation_history) >= 3:
        conversation_history = conversation_history[-3:]

    # Prepare the system message, allowing for non-existent file for testing purposes
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

    #not currently using conversation history context in the prompt because it influences the answers too much
    #prompt_w_context = f"Conversation History:\n{conversation_summary}\n\nUser's Current Question:\n{prompt}" #grab docs related to current prompt + using recent conversation for more context
    
    # Retrieve relevant context from documents
    print('-------------')
    docs = vector_store.similarity_search_with_relevance_scores(prompt, k=3)
    print('document relevance score: ' + str(docs[0][1]))
    context = "\n---\n".join([doc.page_content for doc, _score in docs])
    print("doc used: \n ---" + str(docs[0][0]) + "\n ---")

    # Count and truncate tokens
    context_tokens = count_tokens(context)
    summary_tokens = count_tokens(conversation_summary)
    prompt_tokens = count_tokens(prompt)

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
        print("Prompt: " + prompt)
        print('recent convo hist: \n ---' + conversation_summary + '\n ---')
        print(cb)
        print('\n-------------')
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