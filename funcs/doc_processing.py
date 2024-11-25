import os
import json
from langchain.schema import Document
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

def load_json_to_documents(file_path):
    # Load JSON file
    try:
        with open(file_path, "r") as f:
            json_data = json.load(f)
    except:
        json_data = {}
    documents = []
    
    # Handle the top-level keys like 'person', 'work_experience', 'education', etc.
    # Iterate over each top-level key and process it
    
    # Person information
    person_info = json_data.get('person', {})
    for key, value in person_info.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                page_content = f"{subkey}: {subvalue}"
                documents.append(Document(page_content=page_content, metadata={"category": key}))
        elif isinstance(value, list):  # Lists (e.g., work_experience, hobbies)
            for item in value:
                if isinstance(item, dict):  # If item is a dictionary, concatenate key-value pairs
                    # Convert dictionary into a string
                    page_content = ", ".join(f"{k}: {v}" for k, v in item.items())
                    documents.append(Document(page_content=page_content, metadata={"category": key}))
                else:
                    documents.append(Document(page_content=str(item), metadata={"category": key}))
        else:
            page_content = str(value)
            documents.append(Document(page_content=page_content, metadata={"category": key}))

    # Work experience
    work_experience = person_info.get('work_experience', [])
    for job in work_experience:
        for key, value in job.items():
            if isinstance(value, list):  # If the value is a list (e.g., job_duties)
                # Convert list items into a string, handling possible dictionaries in the list
                page_content = "\n".join(
                    ", ".join(f"{k}: {v}" for k, v in item.items()) if isinstance(item, dict) else str(item)
                    for item in value
                )
                documents.append(Document(page_content=page_content, metadata={"category": f"Work Experience - {job['company']}"}))
            else:
                page_content = str(value)
                documents.append(Document(page_content=page_content, metadata={"category": f"Work Experience - {job['company']}"}))

    # Education
    education = person_info.get('education', {})
    for key, value in education.items():
        page_content = str(value)
        documents.append(Document(page_content=page_content, metadata={"category": "Education"}))

    # Personal interests
    personal_interests = person_info.get('personal_interests', {})
    for key, value in personal_interests.items():
        if isinstance(value, list):  # For hobbies, pets, etc.
            page_content = ", ".join(str(item) if not isinstance(item, dict) else ", ".join(f"{k}: {v}" for k, v in item.items()) for item in value)
            documents.append(Document(page_content=page_content, metadata={"category": "Personal Interests"}))
        else:
            page_content = str(value)
            documents.append(Document(page_content=page_content, metadata={"category": "Personal Interests"}))

    # Social media links
    social_media = person_info.get('social_media_links', {})
    for key, value in social_media.items():
        page_content = f"{key}: {value}"
        documents.append(Document(page_content=page_content, metadata={"category": "Social Media"}))

    print(f"Loaded {len(documents)} documents.")
    return documents

def create_vector_store(documents, persist_directory):
    """Create and persist vector store using Chroma."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    return vector_store