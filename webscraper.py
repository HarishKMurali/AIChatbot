import requests
from bs4 import BeautifulSoup
import re
import sys
import os
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

def scrape_website(base_urls):
    # visited_urls = set()
    scraped_data = []
    for url in base_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the page
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        scraped_data.append((url, text))

    return scraped_data

# Example usage
base_urls = ["https://www.foni.ai/", "https://www.predictiv.ai/", "https://www.predictiv.ai/pages/about-leading-artificial-intelligence-company-toronto", "https://www.predictiv.ai/pages/subsidiaries","https://www.predictiv.ai/pages/investors","https://www.predictiv.ai/blogs/toronto-artificial-intelligence-company-blog"]
scraped_data = scrape_website(base_urls)
print(len(scraped_data))
print(sys.getsizeof(scraped_data))


pc = Pinecone(api_key=PINECONE_API_KEY)


# Initialize Pinecone
# pinecone.init(api_key="d248f58f-0aa6-4952-80e0-809eaaca18ae", environment="us-west1-gcp")
index = pc.Index("predictiv-ai")

# Initialize embedding model (replace with your model of choice)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Embed and store the data
for url, text in scraped_data:
    embedding = embed_text(text)
    # index.upsert_from_dataframe([(url, embedding[0], {"text": text, "url": url})], batch_size=1000)
    index.upsert([(url, embedding[0], {"text": text, "url": url})])

# Don't forget to delete the index after testing
# pinecone.delete_index("predictiv-ai")
