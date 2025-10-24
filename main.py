
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import pandas as pd
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SaveEmb import SaveEmbeddings
from models import *
from load_data import * 
import statistics 
import os



def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_loaders(data, batch_size=6):
    train_loader = NeighborLoader(
        data,
        num_neighbors=[5,2],
        batch_size=batch_size,
        input_nodes=data.train_idx
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=[5, 2],
        batch_size=batch_size,
        input_nodes=data.valid_idx
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=[5,2],
        batch_size=batch_size,
        input_nodes=data.test_idx
    )
    print('type',type(train_loader))
    return train_loader, valid_loader, test_loader

def train_model(model, train_loader, valid_loader, test_loader,dataset_name, epochs=10):
    if dataset_name=='products':
        evaluator = Evaluator(name='ogbn-products')
    else:
        evaluator = Evaluator(name='ogbn-arxiv')

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    train_acc= []
    val_acc= []
    test_acc= []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        y_pred_train = []
        y_true_train = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'): 
            if batch.x.size(0) == 0:
                print("⚠️ Empty batch detected! Skipping this batch.")
                continue

            batch = batch.to(device)
            batch.y = batch.y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch.x, batch.edge_index, batch.n_id, batch.x)

            batch.y = batch.y.squeeze()

            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            preds = outputs.argmax(dim=1)
            y_pred_train.append(preds.cpu())
            y_true_train.append(batch.y.cpu())
        
        y_pred_train = torch.cat(y_pred_train, dim=0)
        y_true_train = torch.cat(y_true_train, dim=0)
        train_result = evaluator.eval({'y_true': y_true_train.unsqueeze(1), 'y_pred': y_pred_train.unsqueeze(1)})
        
        model.eval()
        total_val_loss = 0
        y_pred_val = []
        y_true_val = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{epochs}'): 
            # for batch in valid_loader:
                batch = batch.to(device)
                outputs, _ = model(batch.x, batch.edge_index, batch.n_id, batch.x)
                val_loss = criterion(outputs, batch.y.squeeze())
                total_val_loss += val_loss.item()
                preds = outputs.argmax(dim=1)
                y_pred_val.append(preds.cpu())
                y_true_val.append(batch.y.cpu())

        y_pred_val = torch.cat(y_pred_val, dim=0)
        y_true_val = torch.cat(y_true_val, dim=0)
        y_true_val=y_true_val.squeeze()
        # print('y_pred_val',y_pred_val.shape)
        # print('y_true_val',y_true_val.shape)
        val_result = evaluator.eval({'y_true': y_true_val.unsqueeze(1), 'y_pred': y_pred_val.unsqueeze(1)})

        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs}'): 
            # for batch in test_loader:
                batch = batch.to(device)
                outputs, _ = model(batch.x, batch.edge_index, batch.n_id, batch.x)
                preds = outputs.argmax(dim=1)
                y_pred.append(preds.cpu())
                y_true.append(batch.y.cpu())
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        y_true=y_true.squeeze()
        test_result = evaluator.eval({'y_true': y_true.unsqueeze(1), 'y_pred': y_pred.unsqueeze(1)})
        
        scheduler.step(val_result['acc'])
        print(f"Epoch {epoch+1}: Train Loss: {total_train_loss/len(train_loader):.4f}, "
        f"Train Acc: {train_result['acc']:.4f}, "
        f"Val Loss: {total_val_loss/len(valid_loader):.4f}, "
        f"Val Acc: {val_result['acc']:.4f}, "
        f"Test Accuracy: {test_result['acc']:.4f}")
        train_acc.append(train_result['acc'])
        val_acc.append(val_result['acc'])
        test_acc.append(test_result['acc'])
    max_index = val_acc.index(max(val_acc)) 
    return train_acc[max_index], max(val_acc), test_acc[max_index]

def remap_labels(data):
    """
    Remaps labels to be continuous from 0 to num_classes-1
    """
    unique_labels = torch.unique(data.y)
    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    data.y = torch.tensor([label_map[label.item()] for label in data.y])
    return data, len(unique_labels)
def main():

    parser = argparse.ArgumentParser(description='َ  a hybrid method for node classification')
    

    parser.add_argument('dataset_name',default='arxiv', help='dataset name')
    parser.add_argument('model_name',default='BiGTex', help='dataset name (GCN, GAT, SAGE , BiGTex(ours))')
    parser.add_argument('--batch_size',default=6, type=int, help='size of batch')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs ')
    parser.add_argument('--num_layers', default=2, type=int, help='number of layers ')
    parser.add_argument('--embedding_dim', default=768, type=int, help='size of embeddings ')
    parser.add_argument('--num_iterate', default=10, type=int, help='number of training iteration ')
    parser.add_argument('--language_model_name',default='BERT', help=' (BERT, GPT)')
    parser.add_argument('--soft_prompting',default='True', help=' (True, False)')
    parser.add_argument('--Lora',default='True', help=' (True, False)')
    
    
    args = parser.parse_args()
    epochs=args.epochs
    batch_size=args.batch_size
    num_layers=args.num_layers
    embedding_dim=args.embedding_dim
    model_name=args.model_name
    num_iterate=args.num_iterate
    LM=args.language_model_name
    soft=args.soft_prompting
    Lora=args.Lora



    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name=args.dataset_name
    print("dataset_name= ",dataset_name)
    #------------------------- LOAD DATASET
    if dataset_name=='products':
        data, texts=load_products_subset()
        data, num_classes = remap_labels(data) 
    elif dataset_name=='products':
        data, texts= load_ogb_products()
    elif dataset_name == 'photo':
        data, texts =load_photo()
    elif dataset_name  in ['arxiv', 'arxiv_sim']:
        data, texts=load_arxiv(dataset_name=dataset_name)
    elif dataset_name=='cora':
        data, texts=load_cora()
    elif dataset_name=='pubmed':
        data, texts=load_pubmed()
    elif dataset_name=='arxiv_2023':
        data, texts=load_arxiv_2023()
        data, num_classes = remap_labels(data) 
    #--------------------------------------------------
    data = data.to(device)
    print('y=',len(torch.unique(data.y)))
    #---------------------------- SELECT MODEL
    def select_model(model_name):
        if model_name=="BiGTex":
            model = GraphTextModel(
                feature_dim=data.x.shape[1],
                text_embedding_dim=768,
                embedding_dim=embedding_dim,
                num_classes=len(torch.unique(data.y)),
                texts=texts,
                num_gcn_layers=num_layers, Lora=Lora, soft=soft, LM=LM
            ).to(device)
        elif model_name=="MLP":
            model= SimpleMLP(feature_dim=data.x.shape[1],
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_classes=len(torch.unique(data.y)),
                    dropout=0.2
                    )
        elif model_name=="GT_2":
            model = TextGraphModel(
                embedding_dim=embedding_dim,
                num_classes=len(torch.unique(data.y)),
                texts=texts,
                num_gcn_layers=num_layers
            ).to(device)
        else:
            model=GNN(
                feature_dim=data.x.shape[1],
                GNN_name=model_name,
                num_classes=len(torch.unique(data.y)),
                num_gcn_layers=num_layers
            )
        return model
    #--------------------------------------------------------------------------------
    

    train_loader, valid_loader, test_loader = prepare_loaders(data,batch_size)

    train_acc=[]
    val_acc=[]
    test_acc=[]
    for i in range(num_iterate):
        base_seed=42
        current_seed = base_seed + i  
        set_seed(current_seed)

        print(F"ITERATION {i+1}")
        model = select_model(model_name=model_name)
        # محاسبه تعداد پارامترها
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model_name=",model_name)
        print("LM=", LM)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        train_, val_, test_= train_model(model, train_loader, valid_loader, test_loader,dataset_name, epochs)
        train_acc.append(train_)
        val_acc.append(val_)
        test_acc.append(test_)
        print(f"train{train_acc}, val{val_acc}, test{test_acc}")
        
    
    
     
    print(f"Mean of  Train Acc: {statistics.mean(train_acc):.4f}, "
        f"std of Train Acc: {statistics.stdev(train_acc):.4f}, "
        f"Mean of Val Acc: {statistics.mean(val_acc):.4f}, "
        f"std of Val Acc: {statistics.stdev(val_acc):.4f}, "
        f"Mean of Test Acc: {statistics.mean(test_acc):.4f}, "
        f"std of Test Acc: {statistics.stdev(test_acc):.4f}, ")
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"runtime: {elapsed_time:.4f} sec")  
    print("Save embeddings ...")
    SaveEmbeddings(model,train_loader,valid_loader,test_loader, dataset_name, model_name)
    

if __name__ == '__main__':
    main()

