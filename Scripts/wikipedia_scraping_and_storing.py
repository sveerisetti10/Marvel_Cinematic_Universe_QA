


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

def wiki_summary(url, output_filename):
    # Here we use the requests library to retrieve the webpage
    response = requests.get(url)
    # If the connection was made, then we proceed to extract the content
    if response.status_code == 200:
        # We define the parser, which is the tool used to extract the content
        # We use BeautifulSoup to parse the webpage content
        soup = BeautifulSoup(response.text, 'html.parser')
        # The plot content is usually within the 'mw-parser-output' div tag. We direct the soap object to find this tag
        content_wrapper = soup.find('div', class_='mw-parser-output')
        # Here we create a placeholder for the extracted text
        summary_text = ''
        # If we find the content wrapper, then we proceed to extract the plot
        if content_wrapper:
            # The plot is usually under the 'span' tag with the id 'Plot' or 'Episodes'
            plot_heading = content_wrapper.find('span', id='Plot')
            # If we find either the plot or episodes heading, then we proceed to extract the text
            if plot_heading:
                # We iterate through each element after the plot heading until we find the next section heading
                for elem in plot_heading.parent.find_next_siblings():
                    # If we find another section heading, then we stop
                    if elem.name in ['h2', 'h3']: 
                        break
                    # Here we append the text to the placeholder summary_text
                    summary_text += elem.get_text(separator="\n", strip=True) + '\n\n'
        
        # We can call the clean_text function to clean the extracted text
        summary_text = clean_text(summary_text)
        
        # Finally, we create a file and write the summary to the file
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(summary_text)

        print(f"Summary has been successfully saved to '{output_filename}'")
        return summary_text

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
    url = "https://en.wikipedia.org/wiki/Doctor_Strange_in_the_Multiverse_of_Madness"
    class_names = ['mw-parser-output']
    output_filename = "/Users/sveerisetti/Desktop/Duke_Spring/LLM/Assignments/Assignment2/Scripts/trial/sample_file.txt"
    print("Extracting text from the webpage and storing it in a file")
    summary_text = wiki_summary(url, output_filename)
    print("Storing the chunks within the MongoDB database.")
    store_text_with_embedding(summary_text, url, collection)
