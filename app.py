import os
import concurrent.futures
from functools import lru_cache
import PyPDF2
import logging
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import torch
from torch.utils.data import Dataset
import time

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for the model and tokenizer
embedding_model = None

# Custom Dataset class for fine-tuning
class FineTuneDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Initializes the fine-tuning dataset.
        
        Args:
        - texts (list of str): List of text samples to fine-tune the model on.
        - tokenizer: The tokenizer for encoding the text.
        - max_length (int): Maximum token length for padding/truncating.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns tokenized input and label tensors for a single data sample.
        """
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Labels are set to be identical to the input_ids for language modeling
        inputs['labels'] = inputs['input_ids'].clone()
        
        # Squeeze tensors to remove extra dimensions
        return {k: v.squeeze() for k, v in inputs.items()}


# Function to retrieve relevant content from the FAISS index
def retrieve_data_from_rag(query):
    logging.debug(f"Retrieving relevant content from RAG database for query: {query}")
    
    # Encode the query to obtain its embedding
    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode([query])[0]
    query_embedding = np.array([query_embedding]).astype('float32')
    
    if index.ntotal == 0:
        logging.warning("RAG database is empty. No documents to retrieve.")
        return None
    
    # Search FAISS index for the nearest document to the query
    D, I = index.search(query_embedding, k=1)  # k=1 retrieves the top match
    if I[0][0] == -1:
        logging.warning("No relevant content found in RAG database.")
        return None
    
    # Retrieve the filename of the closest matching document
    matching_doc = document_store[I[0][0]]
    logging.debug(f"Relevant content found in document: {matching_doc['filename']}")
    return matching_doc["filename"]

def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        logging.debug("Loading the embedding model.")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def generate_response_with_model(query, context_filename):
    logging.debug("Generating response with locally installed Granite3 model via Ollama.")
    
    # Prepare the prompt with the user query and relevant document context
    prompt = (
        f"You are an AI assistant. Below is a user query and the name of a relevant document. "
        f"Provide a detailed and helpful response based on the document. "
        f"User Query: {query}\n\n"
        f"Relevant Document: {context_filename}\n\n"
        f"Answer:"
    )
    
    # Use subprocess to run Ollama's Granite3 model
    try:
        start_time = time.time()
        logging.debug(f"Running subprocess with prompt: {prompt}")
        result = subprocess.run(
            ["ollama", "run", "granite3-dense:8b", "prompt", prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',  # Explicitly set encoding to UTF-8 to handle special characters
            timeout=300  # Increase timeout to allow more time for generation
        )
        response = result.stdout.strip() if result.stdout else None
        end_time = time.time()
        logging.debug(f"Response generated successfully in {end_time - start_time:.2f} seconds.")
        
        if not response:
            logging.warning("Generated response is empty. Check prompt or model settings.")
            return "The model did not generate a response. Please try rephrasing your query."
        
        return response
    except UnicodeDecodeError as e:
        logging.error(f"Unicode decoding error: {e}. Please check the encoding settings.")
        return "An encoding error occurred while processing the response. Please try again."
    except subprocess.TimeoutExpired:
        logging.error("Model response generation timed out.")
        return "The model took too long to generate a response. Please try again later."
    except Exception as e:
        logging.error(f"Error generating response with Granite3 model: {e}")
        return "Error generating response."

# Initialize FAISS index and document store only once
logging.debug("Initializing FAISS index.")
embedding_dim = 384  # Dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(embedding_dim)
document_store = []  # To keep track of document metadata

# Functions to load and process files
def extract_text_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return json.dumps(data)

# Function to extract text from a .txt file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to extract text from a .pdf file
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Function to process files and add to FAISS index and document store
def process_file(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    logging.debug(f"Processing file {filename}")

    if os.path.isfile(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        # Extract text based on file extension
        if ext == '.txt':
            text = extract_text_from_txt(file_path)
        elif ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif ext == '.json':
            text = extract_text_from_json(file_path)
        else:
            logging.warning(f"Unsupported file type: {ext}")
            return

        # Add the extracted text to the FAISS index and document store
        if text:
            embedding_model = load_embedding_model()
            embedding = embedding_model.encode([text])[0]
            embedding = np.array([embedding]).astype('float32')
            
            # Add to FAISS index
            index.add(embedding)
            
            # Add to document store
            document_store.append({"filename": filename, "content": text})
            logging.debug(f"File {filename} added to document store and FAISS index.")

# RAG pipeline that retrieves document data and generates a combined response
# RAG pipeline function
def rag_pipeline(user_query):
    logging.debug(f"Starting RAG pipeline for user query: {user_query}")

    # Retrieve relevant document filename
    context_filename = retrieve_data_from_rag(user_query)
    if not context_filename:
        return "No relevant documents found."

    # Generate response with retrieved document filename
    response = generate_response_with_model(user_query, context_filename)

    logging.debug("RAG pipeline completed, returning response.")
    return f"Model Response:\n{response}"

# Example for running file processing without reloading models
if __name__ == "__main__":
    folder_path = "rawData"
    if os.path.exists(folder_path):
        file_names = os.listdir(folder_path)
        for filename in file_names:
            process_file(folder_path, filename)

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        logging.debug(f"User query received: {user_query}")
        response = rag_pipeline(user_query)
        print(response)
