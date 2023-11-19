# combined_app.py

from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import pandas as pd
from googlesearch import search

app = Flask(__name__)

# Load your model and other necessary components
model_path = 'assets\\toxic_comment_classification_model10.h5'
model = load_model(model_path)

# Load and preprocess your dataset
file_path = 'datasets\\train.csv'
df = pd.read_csv(file_path)

x = df['comment_text']
y = df[df.columns[2:]].values

Max_features = 200000  # num of words

vectorizer = TextVectorization(
    max_tokens=Max_features,
    output_sequence_length=1800,
    output_mode='int'
)

vectorizer.adapt(x.values)

def perform_web_scraping(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        body_text = soup.find('body').get_text()
        return body_text
    else:
        return "Error: Unable to fetch the web page content."

def get_top_urls(query, num_results=10):
    try:
        search_results = search(query, num_results=num_results)
        urls = list(set(search_results))
        return urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        keyword = request.form['keyword']
        url = request.form['url']
        results = []

        # Feature 1: URL Directly
        if url:
            try:
                input_texttt = perform_web_scraping(url)
                input_text = vectorizer([input_texttt])
                prediction = model.predict(input_text)
                column_names = df.columns[2:]
                prediction_df = pd.DataFrame(prediction, columns=column_names)
                result = "toxic" if (prediction_df > 0.5).any().any() else "non-toxic"

                results.append({'type': 'URL Directly', 'result': result, 'prediction_df': prediction_df.to_html(), 'input': url})

            except Exception as e:
                print(f"Error processing URL {url}: {e}")

        # Feature 2: Keyword Search
        elif keyword:
            urls = get_top_urls(keyword, num_results=10)

            for url in urls:
                try:
                    if 'facebook.com' in str(url) or 'lybrate.com' in str(url) or "baby360.in" in str(url) or "myupchar.com" in str(url):
                        pass
                    else:
                        input_texttt = perform_web_scraping(url)
                        input_text = vectorizer([input_texttt])
                        prediction = model.predict(input_text)
                        column_names = df.columns[2:]
                        prediction_df = pd.DataFrame(prediction, columns=column_names)
                        result = "toxic" if (prediction_df > 0.5).any().any() else "non-toxic"

                        results.append({'type': 'Keyword Search', 'result': result, 'prediction_df': prediction_df.to_html(), 'input': url})

                except Exception as e:
                    print(f"Error processing URL {url}: {e}")

        return render_template('multi_results.html', results=results, keyword=keyword)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
