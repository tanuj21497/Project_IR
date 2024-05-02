import pandas as pd
from PyPDF2 import PdfReader
import re

import numpy as np
import nltk
import string
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import math
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_pdf(text):
    if isinstance(text, str):
        text = text.replace('\\n', '')  
        text = text.replace('\\t', '')  
        text = text.replace('\\uf06c', '') 
        text = text.lower()

        website_pattern = r'https?://\S+|www\.\S+'

        website_links = re.findall(website_pattern, text)

        placeholder = '###WEBSITE_LINK###'
        text_with_placeholders = re.sub(website_pattern, placeholder, text)

        cleaned_text = text_with_placeholders.replace('/', '')

        for link in website_links:
            cleaned_text = cleaned_text.replace(placeholder, link, 1)

        tokens = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        tokens = [token for token in tokens if token not in string.punctuation]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    else:
        return []

def split_text_into_chunks(text, words=500, words_overlap=100):
    chunks = []
    start = 0
    end = words
    tokens = word_tokenize(text)

    while start < len(text):
        chunk = tokens[start:end]
        if(chunk != []):
          chunks.append(" ".join(chunk))

        start = end - words_overlap
        end = start + words
    return chunks



errors = []
Chunks_df = pd.DataFrame(columns=['SourceFilePath', 'ChunkData'])
count = 0
text_files_dir = "./RE_Download"

# List to store text file paths
text_file_paths = []

# Loop through files in the directory
for file_name in os.listdir(text_files_dir):
    if file_name.endswith('.txt'):
        text_file_path = os.path.join(text_files_dir, file_name)
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            content = text_file.read()

        new_df_rows = []

        # Preprocess the PDF content and generate chunks
        tokens = preprocess_pdf(content)
        data = ' '.join(tokens)

        chunks = split_text_into_chunks(data)

        # Create new rows for each chunk
        for chunk in chunks:
            new_df_rows.append({

                'SourceFilePath': text_file_path,
                'ChunkData': chunk
            })

        chunk_df = pd.DataFrame(new_df_rows)
        Chunks_df = pd.concat([Chunks_df, chunk_df], ignore_index=True)



# Create a new DataFrame from the list of new rows

Chunks_df.to_csv("Chunks.csv")

