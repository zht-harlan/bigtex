import numpy as np
import torch
import random
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_edge_index
import json
from torch_geometric.utils import degree
import os




def load_cora():
    cora_data = torch.load('datasets/cora/cora.pt')
    texts = [
        f"[sep] {text.split(':', 1)[0].strip()} [sep] {text.split(':', 1)[1].strip()}"
        if ':' in text else text
        for text in cora_data.raw_texts
    ]
    #----------------- CORA
    data_name='cora'
    dataset = Planetoid('dataset', data_name,
                            transform=T.NormalizeFeatures())
    data = dataset[0]
    
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=dataset.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=data.train_mask,
        valid_idx=data.val_mask,
        test_idx=data.test_mask,
        )
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_idx = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(data.num_nodes * 0.8):])

    # #  -- splid 60-20-20
    # node_id = np.arange(data.num_nodes)
    # np.random.shuffle(node_id)

    # data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    # data.val_id = np.sort(
    #     node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    # data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    # data.train_mask = torch.tensor(
    #     [x in data.train_id for x in range(data.num_nodes)])
    # data.val_mask = torch.tensor(
    #     [x in data.val_id for x in range(data.num_nodes)])
    # data.test_mask = torch.tensor(
    #     [x in data.test_id for x in range(data.num_nodes)])
    edge_index = data.edge_index
    # degrees = degree(edge_index.view(-1), num_nodes=num_nodes)
    degrees = degree(edge_index.reshape(-1), num_nodes=data.num_nodes)

    avg_degree = degrees.mean().item()
    print(f"Average node degree (undirected): {avg_degree:.4f}")
    # print("data=",data)
    return data, texts

def load_arxiv(dataset_name):
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv')
    data = dataset[0]
    # print("data=",data)
    idx_splits = dataset.get_idx_split()

    if dataset_name=='arxiv_sim':
            x = torch.load('datasets/arxiv_sim/x_embs.pt')
            x=x
    else:
            x=data.x
    
    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=x,  # ویژگی‌های گره‌ها
        edge_index= data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=idx_splits['train'],
        valid_idx=idx_splits['valid'],
        test_idx=idx_splits['test'],
        )

    nodeidx2paperid = pd.read_csv(
        'datasets/arxiv/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('datasets/arxiv/titleabs.tsv.gz',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(str)
    raw_text['paper id'] = raw_text['paper id'].astype(str)
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    texts = []
    for ti, ab in zip(df['title'], df['abs']):
        t = '[sep] ' + ti + '[sep]' + ab
        texts.append(t)

    return data, texts


def load_ogb_products():
    dataset = PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]
    idx_splits = dataset.get_idx_split()
    
    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index= data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=idx_splits['train'],
        valid_idx=idx_splits['valid'],
        test_idx=idx_splits['test'],
        )


    data_root="datasets/products"
    raw_text_path="datasets/products/products_text"
    
    if not os.path.exists(f"{data_root}/product3.csv"):
        i = 1
        for root, dirs, files in os.walk(os.path.join(raw_text_path, '')):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r', encoding='utf-8-sig') as file_in:
                    title = []
                    for line in file_in.readlines():
                        # print("line=",line)
                        dic = json.loads(line)
                        

                        dic['title'] = dic['title'].strip("\"\n")
                        title.append(dic)
                    print("read...")
                    print("len=",len(title))
                    name_attribute = ["uid", "title", "content"]
                    writercsv = pd.DataFrame(columns=name_attribute, data=title)
                    writercsv.to_csv(os.path.join(data_root, f'product' + str(i) + '.csv'), index=False,
                                        encoding='utf_8_sig')  # index=False不输出索引值
                    i = i + 1
        
        pro1 = pd.read_csv(data_root+"/product1.csv")
        pro2 = pd.read_csv(data_root+"/product2.csv")
        file = pd.concat([pro1, pro2])
        file.drop_duplicates()
        file.to_csv(os.path.join(data_root, f'product3.csv'), index=False, sep=" ")
    else:
        file = pd.read_csv(data_root+"/product3.csv", sep=" ")



    category_path_csv = "dataset\ogbn_products\mapping/labelidx2productcategory.csv.gz"
    products_asin_path_csv = "dataset\ogbn_products\mapping/nodeidx2asin.csv.gz"  #
    products_ids = pd.read_csv(products_asin_path_csv)
    categories = pd.read_csv(category_path_csv)

    products_ids.columns = ["ID", "asin"]
    categories.columns = ["label_idx", "category"]  # 指定ID 和 category列写进去
    file.columns = ['asin', 'title', 'content']
    products_ids["label_idx"] = data.y
    data1 = pd.merge(products_ids, file, how="left", on="asin")  # ID ASIN TITLE
    data1 = pd.merge(data1, categories, how="left", on="label_idx")  # 改写是为了拼接到一起

    texts = ('[sep] '+ data1['title'].fillna('') + '[sep] ' + data1['content'].fillna('')).tolist()
    print(len(texts))

    return data, texts

    






# load_ogb_products()
#----------------------------------------------------------------------------------------------------
def load_product():
    dataset = PygNodePropPredDataset(
    name='ogbn-products', transform=T.ToSparseTensor())
    data = dataset[0]
    
    idx_splits = dataset.get_idx_split()

    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index= data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=idx_splits['train'],
        valid_idx=idx_splits['valid'],
        test_idx=idx_splits['test'],
        )
    print("data=",data)
    # data = torch.load('dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('datasets/products/ogbn-products_subset.csv')
    texts = [f'[sep] {ti}. [sep] {cont}'for ti,
            cont in zip(text['title'], text['content'])]
    print("text=",texts[0])
    return data, texts


def load_products_subset():

    data = torch.load('datasets/products/ogbn-products_subset.pt')
    node_desc = pd.read_csv('datasets/products/ogbn-products_subset.csv')

    train_mask, val_mask, test_mask = data.train_mask.squeeze(), data.val_mask.squeeze(), data.test_mask.squeeze()
    
    texts = []
    for i in range(data.num_nodes):
            node_title = (node_desc.iloc[i, 2] if node_desc.iloc[i, 2] is not np.nan else "missing")
            node_content = (node_desc.iloc[i, 3] if node_desc.iloc[i, 3] is not np.nan else "missing")
            text = "[sep] " + node_title + " [sep] " + node_content
            texts.append(text)
    edge_index = data.adj_t.to_symmetric()
    edge_index = to_edge_index(edge_index)[0]
    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index= edge_index,  # یال‌ها
        y=data.y,  # برچسب‌های گره‌ها
        train_idx=torch.where(data.train_mask)[0],
        valid_idx=torch.where(data.val_mask)[0],
        test_idx=torch.where(data.test_mask)[0],
        )
    return data, texts






def load_pubmed():
    
    dataset = Planetoid('dataset', 'PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    print("data",data)

    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=dataset.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        # train_idx=torch.where(data.train_mask)[0],
        # valid_idx=torch.where(data.val_mask)[0],
        # test_idx=torch.where(data.test_mask)[0],
        )
    print("data",data)
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_idx = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(data.num_nodes * 0.8):])

    f = open('datasets/PubMed/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    texts = []
    for ti, ab in zip(TI, AB):
        t = '[sep] ' + ti + '[sep] ' + ab
        texts.append(t)
    print("text",texts[0])



    return data, texts


def load_arxiv_2023():
    data = torch.load('datasets/arxiv_2023/graph.pt')
    print("data",data)
    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=data.y,  # برچسب‌های گره‌ها
        # train_idx=data.train_id,
        # valid_idx=data.val_id,
        # test_idx=data.test_id,
        )
    print("data",data)
    data.train_idx = np.sort(node_id[:int(num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8):])



        # محاسبه میانگین درجه نودها برای گراف بدون جهت
    edge_index = data.edge_index
    # degrees = degree(edge_index.view(-1), num_nodes=num_nodes)
    degrees = degree(edge_index.reshape(-1), num_nodes=num_nodes)

    avg_degree = degrees.mean().item()
    print(f"Average node degree (undirected): {avg_degree:.4f}")

    # data.train_mask = torch.tensor(
    #     [x in data.train_id for x in range(num_nodes)])
    # data.val_mask = torch.tensor(
    #     [x in data.val_id for x in range(num_nodes)])
    # data.test_mask = torch.tensor(
    #     [x in data.test_id for x in range(num_nodes)])

    df = pd.read_csv('datasets/arxiv_2023/paper_info.csv')
    texts = []
    for ti, ab in zip(df['title'], df['abstract']):
        texts.append(f'[sep] {ti}[sep] {ab}')
    return data, texts
#--------------------------photo
from torch_geometric.utils import add_self_loops, to_undirected
def load_photo():
    data = torch.load(f"datasets/photo/photo.pt")
    data.y = data.label
    data.x = data.x.float() # Half into Float
    edge_index = to_undirected(data.edge_index)
    # edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = edge_index
    texts=[]
    for ti in data.raw_texts:
        t = '[sep] ' + ti 
        texts.append(t)
 
    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=data.y,  # برچسب‌های گره‌ها
        )
    # print("data",data)
    data.train_idx = np.sort(node_id[:int(num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8):])



        # محاسبه میانگین درجه نودها برای گراف بدون جهت
    edge_index = data.edge_index
    # degrees = degree(edge_index.view(-1), num_nodes=num_nodes)
    degrees = degree(edge_index.reshape(-1), num_nodes=num_nodes)

    avg_degree = degrees.mean().item()
    print(f"Average node degree (undirected): {avg_degree:.4f}")
    
    # print("data",data)
    return data, texts


def load_citeseer():
    data = torch.load(f"datasets/citeseer/citeseer_random_sbert.pt", weights_only=False)
    print("data", data)
    # print("lanel=", data.label[0])
    # print("y=",data.raw_texts[0])
    # data.y = data.label
    data.x = data.x.float() # Half into Float
    edge_index = to_undirected(data.edge_index)
    # edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = edge_index
    texts=[]
    title=[]
    labels = data.y.numpy() if torch.is_tensor(data.y) else data.y
    print("Unique labels:", np.unique(labels))
    print("Num classes:", len(np.unique(labels)))

    # print("data.raw_texts=",data.raw_texts[0])
    for ti in data.raw_texts:
        
        t = '[sep] ' + ti 
        texts.append(t)
 
    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=data.y,  # برچسب‌های گره‌ها
        )
    # print("data",data)
    data.train_idx = np.sort(node_id[:int(num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8):])



        # محاسبه میانگین درجه نودها برای گراف بدون جهت
    edge_index = data.edge_index
    # degrees = degree(edge_index.view(-1), num_nodes=num_nodes)
    degrees = degree(edge_index.reshape(-1), num_nodes=num_nodes)
    print("data",data)
    avg_degree = degrees.mean().item()
    print(f"Average node degree (undirected): {avg_degree:.4f}")
    
    # print("data",data)
    return data, texts
