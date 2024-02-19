#Overview
The Marvel Universe Data Project is designed to gather, process, and make accessible detailed information about the MCU's movies and TV shows. It provides tools for extracting text, storing it in a MongoDB database, and querying the database to find relevant information using cosine similarity. The project also includes a web application for users to interact with the data, powered by Flask.

#Contents
Data Folder
The data folder contains web-scraped information from two sources:

Wikipedia
Marvel Cinematic Universe Wiki - Fandom
The included Marvel movies and shows are:

Doctor Strange in the Multiverse of Madness
Thor: Love and Thunder
Black Panther: Wakanda Forever
Ant-Man and the Wasp: Quantumania
Guardians of the Galaxy Vol. 3
The Marvels
Loki Season 2 (TV Show)
Echo (TV Show)
This folder holds summaries and actor information for the above movies and shows.

Notebooks Folder
This folder contains Jupyter notebooks used for data processing and analysis:

Marvel_Data_Extraction.ipynb: For data gathering from the mentioned sources.
main_Marvel.ipynb: Defines tokenizer, model, MongoDB database configuration, and functions for extracting text, storing chunks in the database, finding relevant chunks based on cosine similarity, generating enhanced prompts with relevant chunks, and generating text based on this information.
A notebook for comparing cosine similarity scores of responses with and without context, showcasing the performance improvement when using the RAG model with GPT-3.5 turbo alongside relevant chunks.
Scripts Folder
Contains Python scripts for data scraping and storage, parallel to the functionality described in Marvel_Data_Extraction.ipynb:

marvel_scraping_and_storing.py: Focuses on extraction from the Marvel Cinematic Universe Wiki - Fandom.
wikipedia_scraping_and_storing.py: Concentrates on extraction from Wikipedia.
Flask App Folder
This folder houses the Flask web application components:

Routes: view.py defines a route handler for the index page, handling both GET and POST requests.
Static: Contains CSS and image files for the frontend.
Templates: Holds the HTML files for the frontend design.
Utils: Includes database.py, responsible for connecting to the MongoDB database, generating embeddings, finding relevant chunks, generating prompts with context, generating text via GPT-3.5 turbo with context, and returning the generated text.
Installation and Usage
(Provide details on how to set up and run the project, including installing dependencies, setting up the database, and starting the Flask app.)

Contributing
(If open to contributions, provide guidelines on how others can contribute to the project.)

License
(Include license information here.)
