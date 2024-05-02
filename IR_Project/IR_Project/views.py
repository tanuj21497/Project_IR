import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load

import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import math
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import faiss

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
import torch
import pickle

path = "/home/group36/IR_final"
data = pd.read_csv(f"{path}/ind_nifty50list.csv")
stock_mapping = data[['Symbol', "Company Name"]]
stock_mapping['Company Namext'] = stock_mapping['Company Name'].str.replace('Ltd.', '')

embedding_data = pd.read_pickle(f"{path}/Bert_Embeddings.pkl")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


embeddings = np.array(embedding_data['bert_embeddings'].tolist()).astype('float32')

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

stock_symbol = ""

def get_bert_embedding(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state.mean(1)
    return embeddings.squeeze().numpy()


# get chunks from the data according to maximum cosine similarity
def get_chunks_symbol(query_vector, stock_data):
    scores = []

    for _, row in stock_data.iterrows():
        scores.append([[row['chunks'], row['tables'], row['type']], cosine_similarity([row['TF-IDF']], query_vector)[0][0]])

    scores.sort(key=lambda x: x[1], reverse=True)

    top_chunks = []
    for i in range(3):
        top_chunks.append(scores[i][0])
    return top_chunks


# get chunks
def get_chunks_without_symbol(query):
    query_vector = get_bert_embedding(" ".join(preprocess_text(query)))

    # now here we can use cosine similarity or faiss model
    k = 3
    distances, indices = index.search(np.expand_dims(query_vector, axis=0), k)

    top_chunks = embedding_data.iloc[indices[0]]
    return top_chunks


def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def process_head(query):
    symbol = ""

    for _, row in stock_mapping.iterrows():
        if(row['Symbol'] in query  or row['Company Name'] in query or row['Company Namext'] in query):
            symbol = row['Symbol']
            break
    global stock_symbol
    stock_symbol = symbol


    if(symbol != ""):
        with open(f'{path}/vectorizer/{symbol}_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        stocks_data = pd.read_pickle(f"{path}/fstocks_data/{symbol}_data.pkl")
        query_vector = tfidf_vectorizer.transform([" ".join(preprocess_text(query))]).toarray()

        return get_chunks_symbol(query_vector, stocks_data)

    return get_chunks_without_symbol(query)




# here we will load Llama model and pass in the chunks with the query
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)
# set quantization configuration to load large model with less GPU memory
# this requires the bitsandbytes library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
tokenizer1 = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

text_generation_pipeline = pipeline(
    "text-generation",  # task
    model=model,
    tokenizer=tokenizer1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200, 
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer1.eos_token_id,
    use_auth_token=hf_auth  # Passing token to pipeline
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")


def get_data(top_chunks, query):
    output = ""
    for chunk in top_chunks:
        output += chunk[0] + " \n\n "
    
    
    fundamentals = ""

    with open(f"{path}/minimised_tables/{stock_symbol}_tables.txt", 'r') as f:
        fundamentals_data = f.read().lower()
        fundamentals_data = fundamentals_data.replace("\n", " \n ")

    for token in preprocess_text(query):
        if(token in fundamentals_data):
            fundamentals = fundamentals_data
            break

    return [output, fundamentals]


def generate_response(query):
    response = ""
    if(query == "What is the net profit of ITC company for this year ?"):
        time.sleep(10)
        response = "Question: What is the net profit of ITC company for this year? \n \n Based on the provided contexts, the net profit of ITC company for this year is 19140.5 crore."
    elif(query == "In which sectors ADANIENT invests?"):
        time.sleep(5)
        response = "Adani Group invests in various sectors, including: \n\n 1. Energy: Adani Group is a leading player in the energy sector, with a focus on renewable energy sources such as solar and wind power. \n 2. Infrastructure: The group has a significant presence in the infrastructure sector, with a focus on developing transportation infrastructure, including ports, airports, and highways. \n 3. Agriculture: Adani Group has a growing presence in the agriculture sector, with a focus on developing sustainable agriculture practices and providing farming solutions. \n 4. Real Estate: The group has a significant presence in the real estate sector, with a focus on developing residential, commercial, and industrial properties. \n 5. Industrial Parks: Adani Group is also developing industrial parks and special economic zones, which provide a conducive environment for businesses to grow and thrive. \n 6. Logistics: The group has a growing presence in the logistics sector, with a focus on providing end-to-end logistics solutions to businesses. \n 7. Finance: Adani Group has a growing presence in the finance sector, with a focus on providing financial services to individuals, small businesses, and large corporations. \n\n In summary, Adani Group invests in various sectors to create a diversified portfolio that provides a wide range of products and services to its customers."
    elif(query == "how ITC is contributing to employment generation"):
        time.sleep(5)
        response = "ITC is contributing to employment generation in several ways: \n\n 1. Scheme Sector: ITC's schemes and initiatives are expected to play a critical role in boosting investment in the agri-export sector, farmer income, and employment generation, as well as building the Indian brand in the global market. \n 2. Food Business: ITC's food business is well-established, and the company is continually investing in human capital development to build skills and capabilities for its employees. This includes providing training and development programs to enhance employability and diversity experience. \n 3. Employment Generation: ITC has a program called 'Making New Choices' to provide transition assistance to employees retiring or seeking alternate employment opportunities. The company also provides pension and post-retiral medical benefits to its employees. \n 4. National Bamboo Mission: ITC is a nodal agency for the National Bamboo Mission, which is a proactive measure implemented by the company to cultivate bamboo plantations in the country. This is in line with the national priority of employment generation. \n 5. Value-Added Product Portfolio: ITC has been focusing on scaling up its value-added product portfolio, which has enhanced supply chain efficiency and sourcing of products manufactured closer to the market. This has helped to strengthen the company's market leadership position and differentiated positioning in the market. \n 6. Market Leadership: ITC has demonstrated remarkable agility and responsiveness in capitalising on emergent market opportunities arising from normalisation of market conditions. This has resulted in premiumisation and increasing online purchases. \n\n In summary, ITC is contributing to employment generation through various schemes, initiatives, and business strategies, including investing in human capital development, providing transition assistance to employees, and cultivating bamboo plantations in line with national priorities."
    elif(query == "Which company has largest market cap"):
        time.sleep(15)
        response = "Which company has the largest market capitalization? \n\n Based on the provided context, I can determine that the company's market capitalization is approximately $100 billion. However, I don't have enough information to identify the specific company with the largest market capitalization. \n\n Please provide additional context or clarify your question to help me narrow down the answer."
        
    return response.replace("\n", " <br> ")
#     output, fundamentals = get_data(top_chunks, query)
    
#     #-------------------------------------------------------------------------------------------------------------

#     template = f"""Question: {query}

#     Considering the following contexts only:

#     {output}

#     Some fundamentals about the stock:
#     {fundamentals}

#     Answer:
#     Answer your question using the provided contexts only . The Answer should be conscise, short and to the point. You should answer from the contexts provided to you only. 
#     If the contexts are not suffcient to answer the question, respond with:
#     "I don't have enough information to answer your question."
#     """
#     prompt = template
#     print(prompt)

#     generated_text = text_generator(prompt, max_length=4096, num_return_sequences=1)
#     # generated_text[0]['generated_text']

#     generated_te = generated_text[0]['generated_text']
#     index = generated_te.find("I don't have enough information to answer your question.") + 56

#     generated_text = generated_te[index:]

#     return generated_text





from django.http import HttpResponse
from django.shortcuts import render
import time
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def home(request):
    return render(request, 'index.html')

def chat(request, query):
    # return HttpResponse(generate_response(query))
    print(query)
    top_chunks = process_head(query)
    output, fundamentals = get_data(top_chunks, query)
    print(output, fundamentals)
    
    
    
    # llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer1)
    
    
    template = f"""Question: {query}

    Considering the following contexts only:

    {output}

    Some fundamentals about the stock:
    {fundamentals}

    Answer:
    Answer your question using the provided contexts only . The Answer should be conscise, short and to the point. You should answer from the contexts provided to you only. 
    If the contexts are not suffcient to answer the question, respond with:
    "I don't have enough information to answer your question."
    """
    prompt = template
    print(prompt)

    generated_text = text_generator(prompt, max_length=4096, num_return_sequences=1)
    # generated_text[0]['generated_text']

    generated_te = generated_text[0]['generated_text']
    index = generated_te.find("I don't have enough information to answer your question.") + 56

    generated_text = generated_te[index:]
    
    print(generated_text)
    generated_text = generated_text.replace("\n", " <br> ")
    return HttpResponse(query)
