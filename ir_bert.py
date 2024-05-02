import numpy as np
import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import math
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm  # Import tqdm

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

    
# Function to generate BERT embeddings
print("hello", flush=True)
def bert_embeddings(dataframe, text_column):
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Ensure model is in evaluation mode

    # Function to encode text and extract embeddings
    def get_bert_embedding(text):
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        embeddings = output.last_hidden_state.mean(1)
        return embeddings.squeeze().numpy()

    # Apply function to the dataframe with tqdm progress bar
    tqdm.pandas(desc="Calculating embeddings")
    dataframe['bert_embeddings'] = dataframe[text_column].progress_apply(get_bert_embedding)
    return dataframe

# Sample data and usage
data = pd.read_csv("Chunks.csv")
data['type'] = "Financial Statements"
df_abv_21 = data[data['Year'].isin(['2021-2022', '2022-2023'])]
df_abv_21.drop(columns=['Unnamed: 0'], inplace=True)

df_abv_21["ChunkData"] = df_abv_21["ChunkData"].astype(str)
df_with_embeddings = bert_embeddings(df_abv_21, "ChunkData")

# Saving the DataFrame with embeddings to a CSV file
try:
    df_with_embeddings.to_csv('/home/group36/IR/final_embedding.csv', index=False)
    print("CSV saved", flush = True)
except Exception as e:
    print("Error saving CSV:", e)
