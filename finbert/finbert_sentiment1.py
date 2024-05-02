# -*- coding: utf-8 -*-

import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd
merged_df=pd.read_csv("/home/group36/IR/finbert/output.csv")

merged_df = merged_df[['DETAILS', 'Sentiment']]

data = merged_df
data.head()

X = data['DETAILS']
y = data['Sentiment']

len(X)

y.value_counts()

data.shape

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


from tqdm import tqdm
import torch
import scipy.special

preds = []
preds_proba = []
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

for x in tqdm(X):
    with torch.no_grad():
        input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
        logits = model(**input_sequence).logits
        scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        sentimentFinbert = max(scores, key=scores.get)
        probabilityFinbert = max(scores.values())
        preds.append(sentimentFinbert)
        preds_proba.append(probabilityFinbert)

Y1 = []
for sentiment1 in y:
    if sentiment1 == '1':
        Y1.append("positive")
    elif sentiment1 == '0':
        Y1.append("neutral")
    else:
        Y1.append("negative")

Y1[0]='negative'

print(f'Accuracy-Score: {accuracy_score(Y1, preds)}')

print(classification_report(Y1, preds))

cm = confusion_matrix(Y1, preds)
cm_matrix = pd.DataFrame(data=cm)
plt.figure(figsize=(10,10))
sns.heatmap(cm_matrix, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Confusion Matrix')
plt.show()