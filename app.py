import torch
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
import os

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get('CLAUDE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Claude API and Pinecone
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index("predictiv-ai")

# Initialize embedding model (replace with your model of choice)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Function to retrieve relevant context from Pinecone
def retrieve_context(query):
    embedding = embed_text(query)
    result = pc_index.query(vector=embedding[0].tolist(), top_k=3, include_metadata=True)
    context = " ".join([match['metadata']['text'] for match in result['matches']])
    return context

# Function to get a response from Claude
def get_claude_response(prompt, context):
    instructions = "You are a representative of Predictive AI. Given the context and a prompt answer to the user as a company representative. Donot "
    full_prompt = f"{HUMAN_PROMPT}{instructions} {context}\n\n{prompt}{AI_PROMPT}"
    response = anthropic_client.completions.create(
        model="claude-2", 
        prompt=full_prompt,
        max_tokens_to_sample=200
    )
    return response.completion.strip()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_text = request.args.get("msg")
    context = retrieve_context(user_text)
    bot_response = get_claude_response(user_text, context)
    return bot_response

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)
