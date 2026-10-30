from dotenv import load_dotenv
import os
from google import genai
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader

load_dotenv()

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)


db = lancedb.connect("./embeddings")
model = get_registry().get("sentence-transformers").create()

class DataStore(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

table = db.create_table("tag_test", schema=DataStore, mode="overwrite")

documents = [
    """
    Personal Profile:
    My name is Mirshad K. I am from Malappuram, Kerala, India.
    I am an aspiring software developer focused on Python backend
    development and full-stack development.
    I completed my higher secondary education in 2023.
    I completed my software development training at Brototype.
    I am interested in building practical applications and
    understanding how software systems work internally.
    """,

    """
    Technical Skills:
    My primary programming language is Python.
    I have experience with Django and Django REST Framework.
    I also work with React.js for frontend development.
    My backend-related technologies include PostgreSQL, Redis,
    Celery, Django Channels, and Daphne.
    I have experience with Docker, Docker Compose, AWS EC2,
    Nginx, Git, and GitHub.
    """,

    """
    Project - Evenxo:
    Evenxo is a full-stack event management platform.
    It was developed using Django, Django REST Framework,
    React, PostgreSQL, Redis, Celery, and Docker.
    It includes real-time chat functionality, Stripe payments,
    and other event management features.
    The application was deployed using AWS and Docker.
    Its live URL is https://evenxo.xyz.
    """,

    """
    Project - MKart:
    MKart is an e-commerce application developed using Django
    and React.
    It includes e-commerce functionality and Razorpay payment
    integration.
    The project was deployed using Render.
    """,
]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
chunk_size=100, chunk_overlap=10,)

for doc in documents:
    chunks = text_splitter.split_text(doc)

    data_list = [{"text": chunk} for chunk in chunks]

    if data_list:
        table.add(data_list)

while True:
    query = input(": ")
    if query.lower() in ("stop","exit"):
        break
    
    relevant_context = table.search(query).limit(2).to_list()
    context_joined = ','.join(context["text"] for context in relevant_context)
    
    response = client.models.generate_content(
        model='models/gemini-3.8-flash',
        config=types.GenerateContentConfig(
        system_instruction="You are an AI Assistant.Only Answer from given Context"),
        contents=f"""User : {query} 
                     Context:
                     {context_joined}
                     Answer : 
                 """
    )
    print(response.text)