import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from load_data import *
import numpy as np 
import pandas as pd 
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_embeddings(filename, data):
    data1 = pd.read_csv(filename, encoding='utf-8', engine='python')
    embeddings = np.array([np.fromstring(embedding, sep=',') for embedding in data1['Embedding']])
    ids = data1['ID'].values
    
    # مرتب‌سازی امبدینگ‌ها
    embeddings_df = pd.DataFrame(embeddings, index=ids)
    embeddings_sorted = embeddings_df.loc[data.n_id.numpy()]
    
    # تبدیل به tensor و نرمال‌سازی
    embeddings_tensor = torch.tensor(embeddings_sorted.values, dtype=torch.float)
    normalized_embeddings = F.normalize(embeddings_tensor, p=2, dim=1)
    
    return normalized_embeddings

def train_test_split(edge_index, test_ratio=0.1):
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    split = int(num_edges * (1 - test_ratio))
    
    train_edges = edge_index[:, perm[:split]]
    test_edges = edge_index[:, perm[split:]]
    return train_edges, test_edges

def generate_negative_edges(edge_index, num_positive_edges, num_nodes):
    """تولید نمونه‌های منفی به تعداد مساوی با نمونه‌های مثبت"""
    neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_positive_edges,
        method='sparse'  # روش کارآمدتر برای نمونه‌گیری
    )
    return neg_edges

def compute_similarity_scores(edges, embeddings):
    """محاسبه شباهت کسینوسی بین گره‌ها"""
    u, v = edges[0], edges[1]
    u_embeddings = embeddings[u]
    v_embeddings = embeddings[v]
    
    # محاسبه شباهت کسینوسی
    scores = F.cosine_similarity(u_embeddings, v_embeddings)
    return scores

def evaluate_link_prediction(test_edges, negative_edges, embeddings):
    """ارزیابی با استفاده از ROC-AUC و MRR"""
    # محاسبه نمرات
    positive_scores = compute_similarity_scores(test_edges, embeddings)
    negative_scores = compute_similarity_scores(negative_edges, embeddings)
    
    # محاسبه ROC-AUC
    labels = torch.cat([torch.ones(len(positive_scores)), 
                       torch.zeros(len(negative_scores))]).cpu().numpy()
    scores = torch.cat([positive_scores, negative_scores]).cpu().numpy()
    roc_auc = roc_auc_score(labels, scores)
    
    # محاسبه MRR
    ranks = []
    for i, pos_score in enumerate(positive_scores):
        all_scores = torch.cat([negative_scores, pos_score.view(1)])
        rank = (all_scores >= pos_score).sum().item()
        ranks.append(1 / rank)
    mrr = np.mean(ranks)
    
    return roc_auc, mrr

# اجرای اصلی
def main():
    # لود دیتاست و امبدینگ‌ها
    dataset_name = "arxiv_2023"
    
    if dataset_name == 'arxiv':
        data, _=load_arxiv(dataset_name=dataset_name)
    elif dataset_name=='arxiv_2023':
        data, _=load_arxiv_2023()
    
    # data = Data(
    #     n_id=dataset.node_map,
    #     x=dataset.x_original,
    #     edge_index=dataset.edge_index,
    #     y=dataset.label_map,
    # )
    # x='ogb'
    # x='GT'
    # x='simteg'
    x='tape'
    # لود و پردازش امبدینگ‌ها
    if     x == "GT":
        if dataset_name == "arxiv":
            filename = 'embeddings/embeddings_arxiv_GT_SAge_fine.csv'
        elif dataset_name == "arxiv_2023":
            filename = 'embeddings/embeddings_arxiv_2023_GT_SAGE_FINE.csv'
            # data1 = pd.read_csv(filename)
            # embeddings = np.array([np.fromstring(embedding, sep=',') for embedding in data1['Embedding']]) 
            # print(embeddings.shape)
        embeddings = load_and_preprocess_embeddings(filename, data).to(device)
    elif x=='simteg':
        if dataset_name == "arxiv":
            path = 'embeddings/simteg_arxiv.pt'  # یا مسیر کامل فایل
        elif dataset_name == "arxiv_2023":
            path = 'embeddings/sim_arxiv_2023.pt'
        embeddings = torch.load(path)
    elif x=='tape':
        if dataset_name == "arxiv":
            embedding_dim = 384
            emb = np.fromfile("embeddings/tape_arxiv.emb", dtype=np.float32)
            emb = emb.reshape(-1, embedding_dim)
            embeddings = torch.tensor(emb)
        elif dataset_name == "arxiv_2023":
            embedding_dim = 384
            emb = np.fromfile("embeddings/tape_arxiv2023.emb", dtype=np.float32)
            emb = emb.reshape(-1, embedding_dim)
            embeddings = torch.tensor(emb)
    else:
        embeddings=data.x
    
    
    # تقسیم داده‌ها
    train_edges, test_edges = train_test_split(data.edge_index)
    
    # تولید نمونه‌های منفی به تعداد مساوی با تست
    negative_edges = generate_negative_edges(
        train_edges, 
        num_positive_edges=test_edges.size(1),
        num_nodes=data.x.size(0)
    )
    
    # ارزیابی
    roc_auc, mrr = evaluate_link_prediction(test_edges, negative_edges, embeddings)
    print(f"ROC-AUC: {roc_auc:.4f}")
    # print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

if __name__ == "__main__":
    main()