

import requests
from bs4 import BeautifulSoup
import re
import certifi
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import torch 

def clean_text(text):
    """
    Purpose: Clean the extracted text from the Wikipedia page
    text: The text to clean
    """
    # Here we want to remove any text within square brackets, as it's often used for annotations
    text = re.sub(r'\[\w\]', '', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Here we return the text with leading and trailing whitespaces removed
    return text

def extract_sections(url, output_filename, class_names):
    try:
        # Here we use the requests library to retrieve the webpage
        response = requests.get(url)
        # Raise an exception if the status code is not 200
        response.raise_for_status() 
        # Define the parser, which is the tool used to extract the content. In this case we use BeautifulSoup to parse the webpage content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Placeholder for the extracted text
        summary_text = ''

        # Iterate through each class name and extract the content
        for class_name in class_names:
            # Find the content under the class name
            content = soup.find('div', class_=class_name)
            # If we find the content, then we append it to the summary_text
            if content:
                # We can then append the text to the placeholder summary_text
                summary_text += content.get_text(separator="\n", strip=True) + '\n\n'
        
        # We can use the clean_text function to clean the extracted text
        summary_text = clean_text(summary_text)
        # Finally, we create a file and write the summary to the file
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(summary_text)
        # Print a success message
        print(f"Content under classes {class_names} has been successfully saved to '{output_filename}'")
        # Here we will store the text with its embedding in MongoDB
        store_text_with_embedding(summary_text, url, collection)
    except requests.HTTPError as e:
        # If an HTTP error occurs, then we print the error message
        print(f"Failed to retrieve the webpage. HTTP Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_database():
    """
    Purpose: To get the MongoDB database that we will use to store the data.
    """
    # This is the connection to the MongoDB database on its Atlas platform that we are using to store the data
    uri = "mongodb+srv://sveerisetti:HRsAs1@assignment1cluster.lkuwikx.mongodb.net/?retryWrites=true&w=majority"
    ca = certifi.where()
    # Here we create the client that will be used to connect to the MongoDB database
    client = MongoClient(uri, tlsCAFile=ca)
    # Here we select the database and the collection that we want to use
    db = client['trial']
    return db

# Here we utilize the Bert tokenizer in order to tokenize the input text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Here we utilize the Sentence Transformer model to understand the semantic meaning of the input text. This will come in 
# handy when we want to find the most similar sentence to a given input sentence
model = SentenceTransformer('all-mpnet-base-v2')

# This is the key for the OpenAI API that we will use to generate the responses to the user's questions
openai.api_key = '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'

# This function will take in a string and return the embeddings for the string. 
# Here we create a vector of embeddings for the input text
def generate_embedding(text):
    return model.encode(text)

def store_text_with_embedding(text, source, collection):
    if not text:
        return  
    # Here we define a function to chunk the text into smaller pieces with a size of 100
    def chunk_words(text, chunk_size=100):
        # Here we split the text into words
        words = text.split()
        # Here we iterate through the words and yield a chunk of words with the specified size
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i+chunk_size])
    
    # Here we iterate through each chunk of the text and store the chunk along with its embedding in the MongoDB database
    for chunk in chunk_words(text):
        chunk_embedding = generate_embedding(chunk).tolist()
        # We use the insert_one function to insert the chunk and its embedding into the MongoDB database
        collection.insert_one({
            "chunk": chunk,
            "embedding": chunk_embedding,
            "source": source
        })
    print(f"Content from {source} has been successfully stored in MongoDB.")


if __name__ == "__main__":
    # Here we define the database, collection, and url that we want to use. 
    # We also define the place we want to store the txt file that contains the extracted text
    db = get_database()
    collection = db['trial1']
    url = "https://marvelcinematicuniverse.fandom.com/wiki/Doctor_Strange_(film)"
    class_names = ['mw-parser-output']
    output_filename = "/Users/sveerisetti/Desktop/Duke_Spring/LLM/Assignments/Assignment2/Scripts/trial/sample_file.txt"
    print("Extracting text from the webpage and storing it in a file")
    summary_text = extract_sections(url, output_filename, class_names)
    print("Storing the chunks within the MongoDB database.")
    store_text_with_embedding(summary_text, url, collection)
