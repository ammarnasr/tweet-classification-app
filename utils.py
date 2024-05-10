import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from openai import OpenAI
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset



def embeddings_from_tweet(tweets, device):
    client = OpenAI(api_key="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")
    response = client.embeddings.create(
        input=tweets,
        model="text-embedding-3-large"
    )
    embs = []
    for i in range(len(tweets)):
        embs.append(response.data[i].embedding)
    return torch.tensor(embs, dtype=torch.float32, device=device)

class TweetsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TweetClassiferMultiClass(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) # input dim -> 512
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim//2) # 512 -> 256
        self.fc3 = nn.Linear(self.hidden_dim//2, self.hidden_dim//4) # 256 -> 128
        self.fc4 = nn.Linear(self.hidden_dim//4, self.hidden_dim//8) # 128 -> 64
        self.fc5 = nn.Linear(self.hidden_dim//8, self.num_classes) # 64 -> num_classe
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        logits = self.fc5(x)
        x = self.softmax(logits)
        return {'logits': logits, 'probs': x}


def merge_pro_anti(df, pro_col, anti_col):
    pro = df[pro_col].values.tolist()
    anti = df[anti_col].values.tolist()
    merged = []
    for i in range(len(pro)):
        if pro[i] == 1 and anti[i] == 0:
            merged.append(1)
        elif pro[i] == 0 and anti[i] == 1:
            merged.append(0)
        elif pro[i] == 0 and anti[i] == 0:
            merged.append(2)
        else:
            print(f' row {i} has both pro and anti')
            merged.append(2)
    return merged


def get_tweets_embeddings(codes, embs, device):
    tweets_embeddings = []
    for i in range(len(codes)):
        code = codes[i]
        row = embs.loc[code]
        tweets_embeddings.append(row['embedding'])
    tweets_embeddings = torch.tensor(tweets_embeddings, device=device, dtype=torch.float32)
    return tweets_embeddings




def read_labelled_data(process_nan, process_news, process_not_sudan):
    df = pd.read_excel('./data/data.xlsx')
    df.at[596, 'Not about Sudan'] = 0
    df.at[680, 'pro RSF'] = 0
    df.at[774, 'Likely bot'] = 0
    df.at[774, 'Likely not a bot'] = 0
    df.at[687, 'anti SAF'] = 0
    permalinks = df['permalink'].values
    codes = [permalink.split('/')[-1] for permalink in permalinks]
    df['code'] = codes
    subset=['anti RSF', 'pro RSF', 'anti SAF', 'pro SAF', 'Pro peace,', 'anti peace', 'Pro War', 'anti war', 'pro civilian', 'anti civilians', 'Sudanese', 'Not Sudanese']
    x = df[subset].sum()
    n1 = len(df)
    if process_nan:
        df = df.dropna(subset=subset)
        df[['user', 'username']] = df[['user', 'username']].fillna('unknown')
        df = df.fillna(0)
    if process_news:
        df = df[df['Not about Sudan'] == 0]
    if process_not_sudan:
        df = df[df['Not Sudanese'] == 0]
    df = df.reset_index(drop=True)
    y = df[subset].sum()
    n2 = len(df)
    return df


def get_xy(df, tweets_col, labels_cols):
    X = df[tweets_col].reset_index(drop=True)
    if labels_cols:
        Y = df[labels_cols].reset_index(drop=True)
    else:
        Y = df.drop(columns=['post']).reset_index(drop=True)
    codes = df['code'].reset_index(drop=True)
    return X, Y, codes


def merge_into_multicalss(df):
    df['RSF'] = merge_pro_anti(df, 'pro RSF', 'anti RSF')
    df['SAF'] = merge_pro_anti(df, 'pro SAF', 'anti SAF')
    df['peace'] = merge_pro_anti(df, 'Pro peace,', 'anti peace')
    df['war'] = merge_pro_anti(df, 'Pro War', 'anti war')
    df['civilians'] = merge_pro_anti(df, 'pro civilian', 'anti civilians')

    return df


def score_model(model, dataloader, multi_calss, device):
    model.eval()
    ground_truth = []
    predictions = []
    with torch.no_grad():
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)['probs']
            ground_truth.append(y)
            predictions.append(y_pred)
    ground_truth = torch.concat(ground_truth).detach().cpu()
    predictions = torch.concat(predictions).detach().cpu()
    gt = [yp.item() for yp in ground_truth]
    preds = [yp.argmax().item() for yp in predictions]
    acc = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average='weighted')
    if multi_calss == 'raise':
        predictions = torch.tensor([yp.argmax().item() for yp in predictions])
    rocauc = roc_auc_score(ground_truth, predictions, multi_class=multi_calss)
    predictions = torch.where(predictions > 0.5, 1, 0).type(torch.float32)
    scores = {
        'rocauc': rocauc,
        'accuracy': acc,
        'f1': f1
    }
    return scores



def train(model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn, num_classes,model_name, device):
    logs = {'train_loss': [], 'val_loss': [], 'train_rocauc':[], 'val_rocauc':[], 'train_acc':[], 'val_acc':[], 'train_f1':[], 'val_f1':[]}
    acc_max = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)['logits']
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        if epoch%5 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for x, y in val_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)['logits']
                    val_loss += loss_fn(y_pred, y)
            multi_class = 'raise' if num_classes == 2 else 'ovo'
            # train_rocauc = score_model(model, train_dataloader, multi_calss=multi_class)['rocauc']
            # val_rocauc = score_model(model, val_dataloader, multi_calss=multi_class)['rocauc']
            # train_acc = score_model(model, train_dataloader, multi_calss=multi_class)['accuracy']
            # val_acc = score_model(model, val_dataloader, multi_calss=multi_class)['accuracy']
            train_scores = score_model(model, train_dataloader, multi_calss=multi_class, device=device)
            val_scores = score_model(model, val_dataloader, multi_calss=multi_class, device=device)
            train_rocauc, val_rocauc = train_scores['rocauc'], val_scores['rocauc']
            train_acc, val_acc = train_scores['accuracy'], val_scores['accuracy']
            train_f1, val_f1 = train_scores['f1'], val_scores['f1']
            if val_acc > acc_max:
                acc_max = val_acc
                torch.save(model.state_dict(), f'./models/best_{model_name}.pth')
                print(f'>>> Best VAL ACC so far: {acc_max} at epoch {epoch} with train aucroc {train_rocauc} , train acc {train_acc} and val rocauc {val_rocauc} saved to ./models/best_{model_name}.pth')
            print(f'Epoch: {epoch} -- Train Loss: {loss.item() :.4f} RocAuc = {train_rocauc*100 :.4f} Acc = {train_acc*100 :.4f}|| Val Loss: {val_loss.item() :.4f} RocAuc = {val_rocauc*100 :.4f} Acc = {val_acc*100 :.4f}')
            logs['train_loss'].append(loss.item())
            logs['val_loss'].append(val_loss.item())
            logs['train_rocauc'].append(train_rocauc)
            logs['val_rocauc'].append(val_rocauc)
            logs['train_acc'].append(train_acc)
            logs['val_acc'].append(val_acc)
            logs['train_f1'].append(train_f1)
            logs['val_f1'].append(val_f1)
    torch.save(model.state_dict(), f'./models/lattest_{model_name}.pth')
    print(f'>>> Finished training {model_name} model and saved to ./models/lattest_{model_name}.pth, final metrics: train aucroc {train_rocauc} , train acc {train_acc} and val acc {val_acc}')
    return model, logs



def multi_class_weights(data, device):
    w = compute_class_weight(class_weight="balanced", classes=np.unique(data), y=data)
    w =torch.tensor(w, dtype=torch.float32).to(device=device)
    return(w)


def predict_from_embedding(indices , embs , labels, labels_num_classes, hidden_dim, embeddings_dim, device):
    all_predictions = {}
    for label, num_classes in labels_num_classes:
        print(f'Predicting {label}')
        x  = embs.detach().cpu().numpy()[indices]
        y = labels[label].to_numpy()[indices]
        split_num = 0
        ckpt_path = f'./models/best_multiclass_{label}_split_{split_num}.pth'
        model = TweetClassiferMultiClass(embeddings_dim, num_classes, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        x= torch.tensor(x, dtype=torch.float32, device=device)
        y_pred = model(x)['probs']
        ground_truth = torch.tensor(y, dtype=torch.float32)
        predictions = y_pred.detach().cpu()
        gt = [yp.item() for yp in ground_truth]
        preds = [yp.argmax().item() for yp in predictions]
        probs = [yp.max().item() for yp in predictions]
        df = pd.DataFrame({'ground_truth': gt, 'predictions': preds, 'probs': probs, 'index': indices})
        all_predictions[label] = df
    return all_predictions


def predict_from_tweet(tweets, labels_num_classes, hidden_dim, embeddings_dim, device):
    all_predictions = {}
    for label, num_classes in labels_num_classes:
        print(f'Predicting {label}')
        x  = embeddings_from_tweet(tweets, device)
        split_num = 0
        ckpt_path = f'./models/best_multiclass_{label}_split_{split_num}.pth'
        model = TweetClassiferMultiClass(embeddings_dim, num_classes, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(torch.load(ckpt_path))
        x= torch.tensor(x, dtype=torch.float32, device=device)
        y_pred = model(x)['probs']
        predictions = y_pred.detach().cpu()
        preds = [yp.argmax().item() for yp in predictions]
        probs = [yp.max().item() for yp in predictions]
        df = pd.DataFrame({'tweet':tweets, 'predictions': preds, 'probs': probs})
        all_predictions[label] = df
    return all_predictions


