# Import required libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load all pre-trained and pre-processed data files
# These files were created during the ML model training phase
popular_df = pickle.load(open('popular.pkl','rb'))      # Contains popular books data
pt = pickle.load(open('pt.pkl','rb'))                    # Pivot table of users vs books
books = pickle.load(open('books.pkl','rb'))              # Complete books dataset
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))  # Cosine similarity matrix

# Create Flask application object
app = Flask(__name__)

# -------------------- HOME PAGE ROUTE --------------------
@app.route('/')
def index():
    # Sending top popular books data to index.html
    return render_template(
        'index.html',
        book_name = list(popular_df['Book-Title'].values),
        author = list(popular_df['Book-Author'].values),
        image = list(popular_df['Image-URL-M'].values),
        votes = list(popular_df['num_ratings'].values),
        rating = list(popular_df['avg_rating'].values)
    )

# -------------------- RECOMMENDATION PAGE UI --------------------
@app.route('/recommend')
def recommend_ui():
    # Loads the recommendation HTML page
    return render_template('recommend.html')

# -------------------- MAIN RECOMMENDATION LOGIC --------------------
@app.route('/recommend_books', methods=['post'])
def recommend():
    # Get the book name entered by user in the form
    user_input = request.form.get('user_input')

    # Find index of the selected book in pivot table
    index = np.where(pt.index == user_input)[0][0]

    # Sort books based on similarity score
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:5]   # Skip first because it is the same book

    data = []

    # Loop through recommended books
    for i in similar_items:
        item = []

        # Fetch book details from dataset
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]

        # Add book name, author, and image
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    # Print recommended books in terminal (for debugging)
    print(data)

    # Send data back to recommend.html page
    return render_template('recommend.html', data=data)

# -------------------- RUN FLASK APPLICATION --------------------
if __name__ == '__main__':
    app.run(debug=True)
