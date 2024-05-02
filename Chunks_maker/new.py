import pandas as pd
import os
import requests
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
import PyPDF2
from bs4 import BeautifulSoup

# Headers and session setup
headers = {
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US, en, q-0.9",
    "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
}
session = requests.Session()
request = session.get("http://www.nseindia.com/", headers=headers, timeout=20)
cookies = dict(request.cookies)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from XML
def extract_text_from_xml(xml_file_path):
    with open(xml_file_path, 'r', encoding='utf-8') as xml_file:
        soup = BeautifulSoup(xml_file, 'xml')
        text = soup.get_text()
    return text

# Load the CSV file
data = pd.read_csv('result1.csv')

# Create output directory if it doesn't exist
output_dir = "./RE_Download"
os.makedirs(output_dir, exist_ok=True)

# Function to download and save PDF file from zip
def extract_pdf_from_zip(zip_content, output_dir):
    with ZipFile(zip_content) as zip_file:
        # Assume there's only one file in the zip
        file_info = zip_file.infolist()[0]
        pdf_content = zip_file.read(file_info)
        # Extract file name without extension
        file_name = os.path.splitext(file_info.filename)[0] + ".pdf"
        file_path = os.path.join(output_dir, file_name)
        # Save PDF file locally
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        return file_path

# List to store data for new CSV
new_data = []

# Iterate over each row (up to 50 rows)
for index, row in tqdm(data.head(50).iterrows(), total=50):
    # Extract URL from the 'ATTACHMENT' column
    attachment_url = row['ATTACHMENT']
    # Skip invalid links
    if attachment_url == '-':
        continue
    try:
        response = session.get(attachment_url, headers=headers, cookies=cookies, timeout=20)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Check if the content is a zip file
        if response.headers.get('content-type') == 'application/zip':
            # Extract PDF from the zip file
            pdf_file_path = extract_pdf_from_zip(BytesIO(response.content), output_dir)
            text = extract_text_from_pdf(pdf_file_path)
            text_file_path = os.path.splitext(pdf_file_path)[0] + ".txt"
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            file_path = pdf_file_path
        else:
            # Extract file name from URL
            file_name = os.path.basename(attachment_url)
            file_path = os.path.join(output_dir, file_name)
            # Save file locally
            with open(file_path, 'wb') as f:
                f.write(response.content)
            text = ""
            if file_name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_name.lower().endswith('.xml'):
                text = extract_text_from_xml(file_path)
            if text:
                text_file_path = os.path.splitext(file_path)[0] + ".txt"
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
    except Exception as e:
        print(f"Error downloading file from {attachment_url}: {e}")
        continue

    # Append data to the list
    new_data.append({
        'SYMBOL': row['SYMBOL'],
        'COMPANY': row['COMPANY NAME'],
        'SUBJECT': row['SUBJECT'],
        'DETAILS': row['DETAILS'],
        'RECIEPT': row['RECEIPT'],
        'ATTACHMENT': row['ATTACHMENT'],
        'FILE_PATH': file_path,
        'TEXT_PATH': text_file_path
    })

# Create DataFrame from new data
new_df = pd.DataFrame(new_data)

# Save the new DataFrame to a CSV file
new_csv_path = os.path.join(output_dir, 'downloaded_files.csv')
new_df.to_csv(new_csv_path, index=False)

print(f"Downloaded files saved to {new_csv_path}")
