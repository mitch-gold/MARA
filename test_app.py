import pytest
from openai import OpenAI
import os
from app import app  # Import your Flask app

@pytest.fixture
def client():
    # Creates a test client to send requests to your app
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200  # Check that the page returns a 200 OK status
    assert b'<!DOCTYPE html>' in response.data  # Check if HTML is returned

def test_get_bot_response(client):
    response = client.get('/get?msg=Hello')
    assert response.status_code == 200  # Ensure the server responds with status 200
    assert len(response.data) > 0  # Check that the response body is not empty

# Make sure the API key is set from the environment
def test_openai():
    OpenAIclient = OpenAI(api_key = os.environ.get('OPENAI_API_KEY'))

    response = OpenAIclient.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Hello World",
        max_tokens=7,
        temperature=0.1
    )
    
    print(response.choices[0].text.strip())