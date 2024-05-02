import string

# Function to read text from TXT files and preprocess
def preprocess_text(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    # text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove newline characters
    # text = text.replace('\n', '')
    
    return text

# List of TXT files
txt_files = ["h.txt"]

# Preprocess text from TXT files
preprocessed_corpus = [preprocess_text(txt_file) for txt_file in txt_files]

# Display preprocessed text
for i, document in enumerate(preprocessed_corpus):
    print(f"Document {i+1}:\n{document}\n")
