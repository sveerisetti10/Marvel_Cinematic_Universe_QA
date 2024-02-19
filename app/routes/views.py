
from flask import render_template, request, redirect, url_for
from app import app
from app.utils.database import get_response_for_query

# Here we define the route handler for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    error = None

    # Here we check if the request method is POST
    # If the request method is POST, then we process the user's query
    if request.method == 'POST':
        # Gather the user's question from the form
        question = request.form.get('question') 
        try:
            # Here we define the answer based on the user's question
            # We use the get_response_for_query function located in database.py to get the answer
            answer = get_response_for_query(question)
        except Exception as e:
            # Error message to display if an exception occurs
            error = str(e)
    
    # Render the index.html template, passing the answer and error to the template if they exist
    return render_template('index.html', answer=answer, error=error)
