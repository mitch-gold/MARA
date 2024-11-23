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
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_openai():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Hello, world!",
        max_tokens=5
    )
    print(response.choices[0].text.strip())