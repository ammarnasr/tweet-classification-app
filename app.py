import torch
import io
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from model import TweetClassifer
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from utils import read_labelled_data, merge_into_multicalss, get_xy, get_tweets_embeddings, TweetClassiferMultiClass, predict_from_embedding, predict_from_tweet
pd.set_option('display.max_columns', None)
sns.set_theme()

def plot_scores(df):
    f1_df = pd.DataFrame({'Label':df['Label'].values, 'F1':df['Train F1'].values, 'Type':'Train'})
    f1_df = pd.concat([f1_df, pd.DataFrame({'Label':df['Label'].values, 'F1':df['Validation F1'].values, 'Type':'Validation'})])
    f1_df['mean'] = f1_df['F1'].apply(lambda x: float(x.split('±')[0]))
    f1_df['std'] = f1_df['F1'].apply(lambda x: float(x.split('±')[1]))
    fig = px.bar(f1_df, x='Label', y='mean', color='Type', barmode='group', error_y='std')
    fig.update_layout(title='F1 Scores', xaxis_title='Label', yaxis_title='F1 Score')
    rocauc_df = pd.DataFrame({'Label':df['Label'].values, 'ROC AUC':df['Train ROC AUC'].values, 'Type':'Train'})
    rocauc_df = pd.concat([rocauc_df, pd.DataFrame({'Label':df['Label'].values, 'ROC AUC':df['Validation ROC AUC'].values, 'Type':'Validation'})])
    rocauc_df['mean'] = rocauc_df['ROC AUC'].apply(lambda x: float(x.split('±')[0]))
    rocauc_df['std'] = rocauc_df['ROC AUC'].apply(lambda x: float(x.split('±')[1]))
    fig_rocauc = px.bar(rocauc_df, x='Label', y='mean', color='Type', barmode='group', error_y='std')
    fig_rocauc.update_layout(title='ROC AUC Scores', xaxis_title='Label', yaxis_title='ROC AUC Score')
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig)
    c2.plotly_chart(fig_rocauc)

def plot_binary_labels_counts(data):
    binary_labels = ['pro RSF', 'anti RSF', 'anti SAF', 'pro SAF', 'pro peace', 'anti peace', 'pro war', 'anti war', 'pro civilian', 'anti civilians',
                 'no polarisation', 'Geopolticis', 'Sudanese', 'Not Sudanese', 'Likely not a bot', 'Not about Sudan']
    data_copy = data.copy()
    data = data[binary_labels]
    data = data.sum()
    data_precentage = data / data.sum()
    data = pd.DataFrame({'Label':data.index, 'Count':data.values, 'Percentage':data_precentage.values})
    data['Percentage'] = data['Percentage'].apply(lambda x: f'{x:.2%}')
    data = data.sort_values('Count', ascending=False)
    fig = px.bar(data, x='Label', y='Count', text='Percentage')
    fig.update_layout(title='Binary Labels Counts', xaxis_title='Label', yaxis_title='Count')
    st.plotly_chart(fig)
    data = data_copy.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    months = data['month'].unique()
    months = sorted(months)
    months = list(months)
    months.append('All')
    c1, c2 = st.columns(2)
    months_radio = c1.radio('Choose a month', months, index=len(months)-1)
    labels_filter = c2.multiselect('Choose labels', binary_labels, binary_labels)
    if months_radio != 'All':
        data = data[data['month'] == months_radio]
        data_per_month = data[binary_labels].sum()
        data_per_month = pd.DataFrame({'Label':data_per_month.index, 'Count':data_per_month.values, 'Percentage':data_per_month.values})
        data_per_month['Percentage'] = data_per_month['Percentage'] / data_per_month['Percentage'].sum()
        data_per_month['Percentage'] = data_per_month['Percentage'].apply(lambda x: f'{x:.2%}')
        # data_per_month = data_per_month.sort_values('Count', ascending=False)
        data_per_month = data_per_month[data_per_month['Label'].isin(labels_filter)]
        fig = px.bar(data_per_month, x='Label', y='Count', text='Percentage')
        fig.update_layout(title='Binary Labels Counts', xaxis_title='Label', yaxis_title='Count')
        st.plotly_chart(fig)
    else:
        data_per_month = data.groupby('month')[binary_labels].sum()
        data_per_month = data_per_month / data_per_month.sum()
        data_per_month = data_per_month.reset_index()
        data_per_month = pd.melt(data_per_month, id_vars='month', value_vars=binary_labels, var_name='Label', value_name='Percentage')
        fig = px.bar(data_per_month, x='month', y='Percentage', color='Label', barmode='group')
        fig.update_layout(title='Binary Labels Counts per Month', xaxis_title='Month', yaxis_title='Percentage')
        st.plotly_chart(fig)
        st.info('Double click on the legend to show/hide ALL labels')
    st.markdown('---')

def classification_results(data):
    st.markdown('<h3>Classification Results Over All samples not Startified</h3>', unsafe_allow_html=True)
    samples_900 = read_labelled_data(process_nan=True, process_news=False, process_not_sudan=False)
    binary_labels = ['pro RSF', 'anti RSF', 'anti SAF', 'pro SAF', 'Pro peace,', 'anti peace', 'Pro War', 'anti war', 'pro civilian', 'anti civilians',
                    'no polarisation', 'Geopolticis', 'Sudanese', 'Not Sudanese', 'Likely bot', 'Likely not a bot', 'Not about Sudan']
    truth = samples_900[binary_labels].values
    binary_labels = ['pro RSF', 'anti RSF', 'anti SAF', 'pro SAF', 'pro peace', 'anti peace', 'pro war', 'anti war', 'pro civilian', 'anti civilians',
                    'no polarisation', 'Geopolticis', 'Sudanese', 'Not Sudanese', 'Likely bot', 'Likely not a bot', 'Not about Sudan']
    samples_codes = samples_900['code'].values
    data = data.set_index('code')
    data = data.loc[samples_codes]
    preds = data[binary_labels].values
    for i, label in enumerate(binary_labels):
        st.write(f'Label: {label}')
        st.write(f'Accuracy: {accuracy_score(truth[:, i], preds[:, i])}')
        st.write(f'F1 Score: {f1_score(truth[:, i], preds[:, i])}')
        st.write(f'Precision: {precision_score(truth[:, i], preds[:, i])}')
        st.write(f'Recall: {recall_score(truth[:, i], preds[:, i])}')
        st.write(f'ROC AUC: {roc_auc_score(truth[:, i], preds[:, i])}')
        st.markdown('---')



@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8-sig")


def overall_classification():
    only_relevant = st.sidebar.checkbox('Use only relevant tweets', value=False)
    if only_relevant:
        classified_tweets = pd.read_parquet('relevant_classified_tweets.parquet')
    else:
        classified_tweets = pd.read_parquet('./data/combined_reports_with_preds_final.parquet')
    with st.sidebar:
        data_type = 'Relevant' if only_relevant else 'All'
        st.write(f'Downloaded {data_type} classified tweets')
        # csv = convert_df(classified_tweets)
        # st.download_button(
        #     label="Download data as CSV",
        #     data=csv,
        #     file_name=f"classified_tweets_{data_type}.csv",
        #     mime="text/csv",
        #     type='primary',
        #     use_container_width=True
        # )
        # buffer = io.BytesIO()
        # # download button 2 to download dataframe as xlsx
        # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        #     # Write each dataframe to a different worksheet.
        #     classified_tweets.to_excel(writer, sheet_name='Sheet1', index=False)
        #     download2 = st.download_button(
        #         label="Download data as Excel",
        #         data=buffer,
        #         file_name='large_df.xlsx',
        #         mime='application/vnd.ms-excel'
            # )
    stratified_scores = pd.read_csv('./stratified_scores.csv')
    st.markdown('<h3>Stratified Scores</h3>', unsafe_allow_html=True)
    st.write(stratified_scores)
    plot_scores(stratified_scores)
    st.markdown('---')
    cols = ['post','date' ,'pro RSF', 'anti RSF', 'anti SAF', 'pro SAF', 'pro peace', 'anti peace', 'pro war', 'anti war', 'pro civilian', 'anti civilians',
                 'no polarisation', 'Geopolticis', 'Sudanese', 'Not Sudanese', 'Likely bot', 'Likely not a bot', 'Not about Sudan', 'time', 'username', 'code']
    st.markdown('<h3>Classified Tweets</h3>', unsafe_allow_html=True)
    data = classified_tweets[cols]
    plot_binary_labels_counts(data)

    classified_tweets = pd.read_parquet('./data/combined_reports_with_preds_final.parquet')
    data = classified_tweets[cols]
    with st.expander('Show all classified tweets:'):
        classification_results(data)
    




    st.markdown('---')



def tweet_classifier():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    openai_embs = pd.read_parquet('./embeddings/labelled_embeddings.parquet')
    df = read_labelled_data(process_nan=True, process_news=True, process_not_sudan=True)
    df = merge_into_multicalss(df)
    labels = ['RSF','SAF','peace','war','no polarisation','Geopolticis', 'pro civilian', 'anti civilians',
            'Sudanese', 'Not Sudanese', 'Likely bot', 'Likely not a bot', 'Not about Sudan']
    X, Y, codes = get_xy(df, tweets_col='post', labels_cols=labels)
    embs = get_tweets_embeddings(codes, openai_embs, device)
    embeddings_dim = embs.shape[1]
    hidden_dim = 4096
    labels_num_classes = [('RSF', 3), ('SAF', 3), ('peace', 3), ('war', 3), ('no polarisation', 2), ('Geopolticis', 2), ('pro civilian', 2), ('anti civilians', 2),
                        ('Sudanese', 2), ('Not Sudanese', 2), ('Likely bot', 2), ('Likely not a bot', 2), ('Not about Sudan', 2)]
    tweet_src = st.radio('Choose a source for tweets', ['data', 'user'], index=0)
    st.write(f'Classifying tweets using tweets from {tweet_src}')
    if tweet_src == 'data':
        st.write(df[['post', 'RSF', 'SAF', 'peace', 'war', 'civilians', 'no polarisation', 'Geopolticis']])
        tweet_index = st.number_input('Choose a tweet', 0, len(df)-1, 0)
        tweet = df.loc[tweet_index, 'post']
        st.write(tweet)
        s = predict_from_embedding([tweet_index] , embs , Y, labels_num_classes, hidden_dim, embeddings_dim, device)
    else:
        tweet = st.text_area('Enter your tweet', value='Stop the war')
        s = predict_from_tweet([tweet], labels_num_classes, hidden_dim, embeddings_dim, device)
    unmerge_check = st.checkbox('Unmerge', value=False)
    if not unmerge_check:
        data = pd.DataFrame({'key': [], 'ground_truth': [], 'predictions': [], 'probs': []})
        color_map = {True: 'green', False: 'red'}
        target_index = 0
        for key in s.keys():
            preds = s[key].iloc[target_index]
            if 'ground_truth' in preds:
                gt = preds['ground_truth']
            else:
                gt = preds['predictions']
            row = pd.DataFrame({'key': [key], 'ground_truth': [gt], 'predictions': [preds['predictions']], 'probs': [preds['probs']]})
            data = pd.concat([data, row])
        data = data.reset_index(drop=True)
        data['is_correct'] = data['ground_truth'] == data['predictions']
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.barplot(data=data, x='key', y='probs', hue='is_correct', ax=ax, palette=color_map) 
        st.pyplot(fig)
        st.write(data)
    else:
        data_unmerged = pd.DataFrame({'key': [], 'ground_truth': [], 'predictions': [], 'probs': []})
        color_map = {True: 'green', False: 'red'}
        target_index = 0
        for label, num_classes in labels_num_classes:
            preds = s[label].iloc[target_index]
            if 'ground_truth' in preds:
                gt = preds['ground_truth']
            else:
                gt = preds['predictions']
            pred = preds['predictions']
            prob = preds['probs']
            if num_classes == 2:
                row = pd.DataFrame({'key': [label], 'ground_truth': [gt], 'predictions': [pred], 'probs': [prob]})
                data_unmerged = pd.concat([data_unmerged, row])
            else:
                pro_label = f'pro {label}'
                pro_gt = 1 if gt == 1 else 0
                pro_preds = 1 if pred == 1 else 0
                pro_probs = prob if pred == 1 else 1-prob
                row = pd.DataFrame({'key': [pro_label], 'ground_truth': [pro_gt], 'predictions': [pro_preds], 'probs': [pro_probs]})
                data_unmerged = pd.concat([data_unmerged, row])
                anti_label = f'anti {label}'
                anti_gt = 1 if gt == 0 else 0
                anti_preds = 1 if pred == 0 else 0
                anti_probs = prob if pred == 0 else 1-prob
                row = pd.DataFrame({'key': [anti_label], 'ground_truth': [anti_gt], 'predictions': [anti_preds], 'probs': [anti_probs]})
                data_unmerged = pd.concat([data_unmerged, row])
        data_unmerged = data_unmerged.reset_index(drop=True)
        data_unmerged['is_correct'] = data_unmerged['ground_truth'] == data_unmerged['predictions']
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.barplot(data=data_unmerged, x='key', y='probs', hue='is_correct', ax=ax, palette=color_map)
        st.pyplot(fig)
        st.write(data_unmerged)



if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.markdown("<h2 style='text-align: center; color: black;'>Overall Results</h2>", unsafe_allow_html=True)
    overall_classification()
    st.markdown("<h2 style='text-align: center; color: black;'>Classifier Demo</h2>", unsafe_allow_html=True)
    tweet_classifier()