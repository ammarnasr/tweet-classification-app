import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from openai import OpenAI
from tqdm.auto import tqdm
import os
import joblib
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")



class TweetClassifer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) # input dim -> 512
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim//2) # 512 -> 256
        self.fc3 = nn.Linear(self.hidden_dim//2, self.hidden_dim//4) # 256 -> 128
        self.fc4 = nn.Linear(self.hidden_dim//4, self.hidden_dim//8) # 128 -> 64
        self.fc5 = nn.Linear(self.hidden_dim//8, self.num_classes) # 64 -> num_classe

    def forward(self, x, logits=False):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        if logits:
            x = torch.sigmoid(self.fc5(x))
        else:
            x = self.fc5(x)
        return x
    

class TweetsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



def read_data():
    df = pd.read_excel('data.xlsx')
    # Change weired values
    df.at[596, 'Not about Sudan'] = 0
    df.at[680, 'pro RSF'] = 0
    df.at[774, 'Likely bot'] = 0
    df.at[774, 'Likely not a bot'] = 0
    codes = []
    for i in range(len(df)):
        codes.append(f'TW{i :03d}')
    df['code'] = codes
    return df

def get_xy(df, tweets_col = 'post', labels_cols=None):
    X = df[tweets_col]
    if labels_cols:
        Y = df[labels_cols]
    else:
        Y = df.drop(columns=['post'])
    codes = df['code']
    return X, Y, codes

def get_tweets_embeddings(tweets, model_name="all-MiniLM-L6-v2", src = 'hf', codes=None):
    with torch.no_grad():
        if src=='hf':
            batch_size = 1000
            tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-msa')
            model = AutoModel.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-msa')
            if len(tweets) < batch_size:
                encoded_input = tokenizer(tweets, return_tensors='pt', padding=True )
                tweets_embeddings = model(**encoded_input).pooler_output
            else: #Batching
                n = len(tweets)
                tweets_embeddings = []
                for i in tqdm(range(0, len(tweets), batch_size)):
                    batch = tweets[i:i+batch_size]
                    encoded_input = tokenizer(batch, return_tensors='pt', padding=True )
                    batch_embeddings = model(**encoded_input).pooler_output
                    tweets_embeddings.append(batch_embeddings)
                tweets_embeddings = torch.cat(tweets_embeddings)
        elif src=='sentence_transformers':
            embeddings_model = SentenceTransformer(model_name).to(device)
            tweets_embeddings = embeddings_model.encode(tweets, convert_to_tensor=True)
        elif src == 'openai':
            if type(codes) == type(None):
                raise ValueError ('Codes must not be None when using openai as src')
            tweets_embeddings = []
            openai_embs = load_openai_embs()
            openai_embs = openai_embs.set_index('code')
            for i in range(len(codes)):
                code = codes[i]
                og_tweet= tweets[i]
                row = openai_embs.loc[code]
                matched_tweet = row["post"]
                if og_tweet != matched_tweet:
                    raise ValueError(f'Orginal Tweet dont Match found tweet and Code {code}, og_tweet:{og_tweet} -- matched_tweet{matched_tweet}')
                tweets_embeddings.append(row['openai_large_embs'])
            tweets_embeddings = torch.tensor(tweets_embeddings, device=device, dtype=torch.float32)
        else:
            raise ValueError('src must be one of ["hf", "sentence_transformers", "openai"]')

    return tweets_embeddings

def embeddings_pca(embs, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embs)





def score_model(model, dataloader):
    model.eval()
    ground_truth = []
    predictions = []
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, logits=True)
            ground_truth.append(y)
            predictions.append(y_pred)
    ground_truth = torch.concat(ground_truth).detach().cpu()
    predictions = torch.concat(predictions).detach().cpu()
    rocauc = roc_auc_score(ground_truth, predictions)
    predictions = torch.where(predictions > 0.5, 1, 0).type(torch.float32)
    if y_pred.shape[1]>1:
        mcm =  multilabel_confusion_matrix(ground_truth, predictions)
        true_negatives = mcm[:,0,0]
        false_negatives = mcm[:,1,0]
        false_positives = mcm[:,0,1]
        true_positives = mcm[:,1,1]
    else:
        cm = confusion_matrix(ground_truth, predictions)
        true_negatives  = cm[0,0]
        false_negatives = cm[1,0]
        false_positives = cm[0,1]
        true_positives  = cm[1,1]
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    scores = {
        'accuracy': (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives),
        'precision': precision,
        'recall': recall,
        'f1': 2 * (precision * recall) / (precision + recall),
        'rocauc': rocauc
    }
    return scores
    

def calculate_pos_weights(data):
    class_counts = data.sum(axis=0).to_numpy()
    pos_weights = np.ones_like(class_counts)
    neg_counts = [len(data)-pos_count for pos_count in class_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)
    return torch.as_tensor(pos_weights, dtype=torch.float, device=device)




def openai_embedding(c, tweet, model="text-embedding-3-large"):
    response = c.embeddings.create(
        input=tweet,
        model=model   
    )
    return response.data[0].embedding

def save_openai_embs(df):
    client = OpenAI(api_key='')
    large_embs = []
    for i,row in df.iterrows():
        tweet = row['post']
        code = row['code']
        emb = openai_embedding(client, tweet)
        large_embs.append(emb)
    df['openai_large_embs'] = large_embs
    df.to_parquet('openai_large_embs.parquet')
    return df

def load_openai_embs():
    df = pd.read_parquet('openai_large_embs.parquet')
    return df
