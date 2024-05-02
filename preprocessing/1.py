import numpy as np
import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import math

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = ""

with open("h.txt", 'r', encoding='utf-8') as f:
    data = f.read()

def preprocess_text(text):
    # Check if the text is a string
    if isinstance(text, str):
        # Lowercasing
        text = text.replace('\\n', '')  # Remove newline characters
        text = text.replace('\\t', '')  # Remove newline characters
        text = text.replace('\\uf06c', '')  # Remove '\uf06c' characters
        text = text.lower()
        # Regular expression pattern to match website links
        website_pattern = r'https?://\S+|www\.\S+'

        # Find all website links in the text
        website_links = re.findall(website_pattern, text)

        # Replace each website link with a placeholder string
        placeholder = '###WEBSITE_LINK###'
        text_with_placeholders = re.sub(website_pattern, placeholder, text)

        # Remove "/" characters from the text
        cleaned_text = text_with_placeholders.replace('/', '')

    # Re-insert website links back into their original positions
        for link in website_links:
            cleaned_text = cleaned_text.replace(placeholder, link, 1)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]


        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    else:
        return []

filtered_tokens = preprocess_text(data)
da = ' '.join(filtered_tokens)
# print(da)

# Define the path for the new text file
output_file_path = "preprocessed_text.txt"

# Write the preprocessed text to the new text file
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(da)

print("Preprocessed text has been saved to:", output_file_path)