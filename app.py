import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
import streamlit as st
from eda import read_data, get_tweets_embeddings, TweetClassifer
import seaborn as sns
import numpy as np

def sigmoid(x):
    return torch.sigmoid(torch.tensor(x)).item()

def classify_tweets(emb, clf, device):
    with torch.no_grad():
        pred = clf(emb.to(device))
    return pred.cpu().numpy()

def plot_predictions(preds, labels_names, labels=None):
    fig, ax = plt.subplots(figsize=(20, 4))
    preds = np.array([sigmoid(p) for p in preds])
    sns.barplot(x=labels_names, y=preds, ax=ax, hue=labels, palette=['red', 'green'])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='blue', linestyle='--')
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Label')
    plt.legend(title='Ground Truth')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    if labels is not None:
        st.write(pd.DataFrame({'prediction': preds>0.5, 'ground_truth': labels}))

@st.cache_data
def load_clf(input_dim, num_classes, model_path):
    model = TweetClassifer(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

@st.cache_data
def load_emb_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == '__main__':
    st.title('Tweet Classifier')
    emb_src = st.radio('Choose a source for embeddings', ['openai', 'hf'], index=1)
    tweet_src = st.radio('Choose a source for tweets', ['data', 'user'], index=0)
    st.write(f'Classifying tweets using embeddings from {emb_src} and tweets from {tweet_src}')
    labels_names = ['pro RSF', 'anti RSF', 'anti SAF', 'pro SAF', 'Pro peace,', 'anti peace', 'Pro War',
    'anti war', 'pro civilian', 'anti civilians', 'no polarisation', 'Geopolticis', 'Sudanese', 'Not Sudanese']
    if tweet_src == 'data':
        data = read_data()
        st.write(data)
        tweet_index = st.number_input('Choose a tweet', 0, len(data)-1, 0)
        tweet = data.loc[tweet_index, 'post']
        code = data.loc[tweet_index, 'code']
        labels = data.loc[tweet_index, labels_names]
        st.write(tweet)
    else:
        tweet = st.text_area('Enter your tweet')
        code = None
        labels = None
    # classify_btn = st.button('Classify')
    if True:
        with st.spinner('Classifying...'):
            with torch.no_grad():
                with st.empty():
                    st.write('Getting embeddings...')
                    embeddings = get_tweets_embeddings([tweet], codes=[code], src=emb_src)
                    st.write('Loading model...')
                    random_state = 42
                    test_size = 0.3
                    embeddings_dim = embeddings.shape[1]
                    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                    num_classes = len(labels_names)
                    hidden_dim = 1024 if emb_src == 'hf' else 2048
                    model = TweetClassifer(embeddings_dim, num_classes, hidden_dim=hidden_dim).to(device)
                    if emb_src == 'hf':
                        model_path = f'model_{emb_src}_{random_state}_{test_size}.pth'
                    else:
                        model_path = f'lattest_model_{emb_src}_{random_state}_{test_size}.pth'
                    model.load_state_dict(torch.load(model_path))
                    st.write('Classifying...')
                    preds = classify_tweets(embeddings, model, device)[0]
                    st.write(f'Loaded model from {model_path}')
        plot_predictions(preds, labels_names, labels)


