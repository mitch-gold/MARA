import pytest
import openai
import os
from app import app  # Import your Flask app

@pytest.fixture
def client():
    # Creates a test client to send requests to your app
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page (index.html)"""
    response = client.get('/')
    assert response.status_code == 200  # Check if the status code is 200 OK
    assert b'Welcome' in response.data  # Check if the page contains the word 'Welcome'

def test_get_bot_response(client):
    """Test the `/get` endpoint (your chatbot endpoint)"""
    response = client.get('/get?msg=Hello')
    assert response.status_code == 200  # Check if the response status is 200 OK
    assert b'Hello' in response.data  # Check if the response contains the word 'Hello'

# Make sure the API key is set from the environment
def test_openai():
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}]
    )

    print(response['choices'][0]['message']['content'])