
import certifi
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import torch 


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
    db = client['Marvel']
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

def find_most_relevant_chunks(query, top_k=5):
    """
    Purpose: To extract text from .txt files in a directory and store the text along with its embedding in MongoDB.
    directory_path: The path to the directory containing .txt files.
    """
    # Here we get the database using the get_database function
    db = get_database()
    # Here we specify the collection that we want to use
    collection = db['superhero_chunk']
    # We use the generate_embedding function to get the embedding for the input query
    query_embedding = generate_embedding(query)
    # Here we get the documents from the collection
    docs = collection.find({})

    # Here we gather all of the chunks from the MongoDB database and calculate the cosine similarity between the query and each chunk
    # The similarity empty list will be used to store the similarities between the query and each chunk
    similarities = []
    for doc in docs:
        chunk_embedding = doc['embedding']
        # Here we calculate the cosine similarity between the query and the chunk in the MongoDB database. 
        similarity = cosine_similarity([chunk_embedding], [query_embedding])[0][0]
        # Here we append the chunk, similarity, and source to the similarities list
        similarities.append((doc['chunk'], similarity, doc.get('source')))

    # Here sort the similarities list by the similarity score in descending order. 
    # The top will be the most similar chunks to the input query.
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top k most similar chunks, filtering out any duplicates
    seen_chunks = set()
    unique_similarities = []
    # Here we loop through the similarities list and append the unique chunks to the unique_similarities list
    for chunk, similarity, source in similarities:
        # Here we check if the chunk is not in the seen_chunks set
        if chunk not in seen_chunks:
            # If there is a seen chunk, then we add the chunk to the seen_chunks set 
            seen_chunks.add(chunk)
            # Here we only append the unique chunks to the unique_similarities list
            unique_similarities.append((chunk, similarity, source))
            # Here we make sure the returned list is of length top_k
            if len(unique_similarities) == top_k:
                break
    return unique_similarities


def generate_prompt_with_context(relevant_chunks, query):
    """
    Purpose: Generates a prompt with the relevant chunks and the input query.
    relevant_chunks: The most relevant chunks to the input query.
    query: The input query.
    """
    # Here we build context for the prompt by adding the relevant chunks to the prompt
    context = "Based on the following information: "
    # Here we loop through the relevant chunks and add them to the context
    for chunk, similarity, source in relevant_chunks:
        # Here we add the chunk, similarity, and source to the context
        context += f"\n- [Source: {source}, Similarity: {similarity}]: {chunk}"
    # Here we concatenate the context and the query to create the prompt
    prompt = f"{context}\n\n{query}"
    return prompt

def generate_text_with_gpt35(prompt, max_tokens=3100):
    """
    Purpose: Generates a response to the user's query using the GPT-3.5 model.
    prompt: The prompt to generate the response.
    max_tokens: The maximum number of tokens for the response.
    """
    # Here we use the OpenAI API to generate the response to the user's query
    response = openai.ChatCompletion.create(
        # Here we specify the model that we want to use
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        # We define the maximum number of tokens for the response based on the max_tokens parameter
        max_tokens=max_tokens,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].message['content'].strip()


def get_response_for_query(query):
    """
    Purpose: Generates a response to the user's query.
    query: The user's query.
    """
    # Here we find the most relevant chunks to the user's query within the MongoDB database
    # We use the find_most_relevant_chunks function to find the most relevant chunks to the user's query
    relevant_chunks = find_most_relevant_chunks(query)
    # If there are relevant chunks, then we generate a prompt with the relevant chunks and the input query
    if relevant_chunks:
        # We can use the generate_prompt_with_context function to generate a prompt with the relevant chunks and the input query
        prompt = generate_prompt_with_context(relevant_chunks, query)
        # We can then use the generate_text_with_gpt35 function to generate the response to the user's query
        return generate_text_with_gpt35(prompt)
    else:
        return "Sorry, but there is no relevant information available for this query."
