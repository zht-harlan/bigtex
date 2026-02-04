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
from finetune_bert import finetune_bert_with_soft_prompt


def set_seed(seed):
    """تنظیم seed برای تمام عملیات تصادفی"""
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
        num_neighbors=[-1,-1],
        batch_size=batch_size,
        input_nodes=data.train_idx
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=[-1,-1],
        batch_size=batch_size,
        input_nodes=data.valid_idx
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1,-1],
        batch_size=batch_size,
        input_nodes=data.test_idx
    )
    print('type',type(train_loader))
    return train_loader, valid_loader, test_loader

def train_model(model,train_idx, train_loader, valid_loader, test_loader,dataset_name, epochs=10,mode="AE",num_classes=7,alpha=None):
    if dataset_name=='products':
        evaluator = Evaluator(name='ogbn-products')
    else:
        evaluator = Evaluator(name='ogbn-arxiv')
    
    device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss()


    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    train_acc= []
    val_acc= []
    test_acc= []
    best_val_acc = 0
    best_model_state = None
    log_file = f"result_log_{dataset_name}.txt"

    with open(log_file, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,test_acc\n")

    for epoch in range(epochs):
        model.train()
        # model.set_phase('train') 
        total_train_loss = 0
        y_pred_train = []
        y_true_train = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            if batch.x.size(0) == 0:
                print("⚠️ Empty batch detected! Skipping this batch.")
                continue
            
            batch = batch.to(device)
            batch.y = batch.y.to(device)
            batch_size = batch.batch_size
            optimizer.zero_grad()

            outputs,_, _,g_proj, t_proj = model(batch.x, batch.edge_index, batch.n_id, batch.x,batch_size)
            #
            #  only seed nodes
            batch_size = batch.batch_size
            seed_outputs = outputs[:batch_size]

            seed_labels = batch.y.squeeze()
            seed_labels = batch.y[:batch_size]
            if seed_labels.size(0) != batch_size:
                seed_labels = seed_labels[:batch_size]
            
            seed_labels = seed_labels.squeeze().long()
            if seed_outputs.dim() == 0 or seed_labels.dim() == 0:
                    continue
 
            one_hot_labels = F.one_hot(seed_labels, num_classes=num_classes).float()
            loss_task = criterion(seed_outputs, one_hot_labels)


      
            if mode=="AE":
                loss_recon = 0.0
                if hasattr(model, 'last_ae_recon'):
                    loss_recon = F.mse_loss(
                        model.last_ae_recon,
                        model.last_ae_input
                    )
                loss = loss_task + 0.1 * loss_recon
            else:
                loss = loss_task 

 

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            preds = seed_outputs.argmax(dim=1)
            y_pred_train.append(preds.cpu())
            y_true_train.append(seed_labels.cpu())
        
        y_pred_train = torch.cat(y_pred_train, dim=0)
        y_true_train = torch.cat(y_true_train, dim=0)
        train_result = evaluator.eval({'y_true': y_true_train.unsqueeze(1), 'y_pred': y_pred_train.unsqueeze(1)})
        
        model.eval()
        # model.set_phase('val')
        total_val_loss = 0
        y_pred_val = []
        y_true_val = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                batch = batch.to(device)
                
                outputs,_, _,g_proj, t_proj = model(batch.x, batch.edge_index, batch.n_id, batch.x,batch.batch_size)
                
                # ✅ فقط seed nodes
                batch_size = batch.batch_size
                seed_outputs = outputs[:batch_size]

                seed_labels = batch.y.squeeze()
                seed_labels = batch.y[:batch_size]
                
                # ✅ بررسی سایز
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]
                seed_labels = seed_labels.squeeze().long()
                if seed_outputs.dim() == 0 or seed_labels.dim() == 0:
                    continue
                # val_loss = criterion(seed_outputs, seed_labels)
                one_hot_labels = F.one_hot(seed_labels, num_classes=num_classes).float()
                val_loss = criterion(seed_outputs, one_hot_labels)


                total_val_loss += val_loss.item()
                preds = seed_outputs.argmax(dim=1)

                y_pred_val.append(preds.cpu())
                y_true_val.append(seed_labels.cpu())
        
        y_pred_val = torch.cat(y_pred_val, dim=0)
        y_true_val = torch.cat(y_true_val, dim=0)
        y_true_val=y_true_val.squeeze()
        
        val_result = evaluator.eval({'y_true': y_true_val.unsqueeze(1), 'y_pred': y_pred_val.unsqueeze(1)})

        if val_result['acc'] > best_val_acc:
            best_val_acc = val_result['acc']
            best_model_state = model.state_dict()
    
        
        model.eval()
        # model.set_phase('test')
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                batch = batch.to(device)
                
                outputs,_, _,g_proj, t_proj = model(batch.x, batch.edge_index, batch.n_id, batch.x,batch.batch_size)
                
                # ✅ فقط seed nodes
                batch_size = batch.batch_size
                seed_outputs = outputs[:batch_size]
 
                seed_labels = batch.y.squeeze()
                seed_labels = batch.y[:batch_size]
                
                # ✅ 
                if seed_labels.size(0) != batch_size:
                    seed_labels = seed_labels[:batch_size]
                seed_labels = seed_labels.squeeze().long()
                if seed_outputs.dim() == 0 or seed_labels.dim() == 0:
                    continue
   
                preds = seed_outputs.argmax(dim=1)

                y_pred.append(preds.cpu())
                y_true.append(seed_labels.cpu())
        
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
        with open(log_file, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{total_train_loss/len(train_loader):.4f},"
                f"{train_result['acc']:.4f},"
                f"{total_val_loss/len(valid_loader):.4f},"
                f"{val_result['acc']:.4f},"
                f"{test_result['acc']:.4f}\n"
            )

        
        train_acc.append(train_result['acc'])
        val_acc.append(val_result['acc'])
        test_acc.append(test_result['acc'])
    
    max_index = val_acc.index(max(val_acc))

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_name=dataset_name+"_model.pt"
    torch.save(model.state_dict(), model_name)
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

    parser.add_argument('dataset_name',default='cora', help='dataset name')
    parser.add_argument('model_name',default='BiGTex', help=' (GCN, GAT, SAGE , BiGTex(ours))')
    parser.add_argument('--batch_size',default=64, type=int, help='size of batch')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs ')
    parser.add_argument('--num_layers', default=2, type=int, help='number of layers ')
    parser.add_argument('--embedding_dim', default=768, type=int, help='size of embeddings ')
    parser.add_argument('--num_iterate', default=5, type=int, help='number of traaining iteration ')
    parser.add_argument('--language_model_name',default='SCIBERT', help=' (BERT, GPT, SCIBERT, DeBERTA)')
    parser.add_argument('--soft_prompting',default='True', help=' (True, False)')
    parser.add_argument('--Lora',default='True', help=' (True, False)')
    parser.add_argument('--GNN',default='sage', help=' (gcn, gat, sage)')
    parser.add_argument('--finetune_epochs', default=10,type=int, help='number of epochs ')
    parser.add_argument('--finetune_batch_size', default=16,type=int, help='number of finetune_batch_size ')
    parser.add_argument('--mode',default='MLP', help=' (AE, MLP, cross)')
    parser.add_argument('--use_adaptive', default='True', help=' (True, False)')

    
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
    GNN=args.GNN
    mode=args.mode


    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name=args.dataset_name
    #------------------------- LOAD DATASET
    if dataset_name=='products-subset':
        data, texts=load_products_subset()
        data, num_classes = remap_labels(data)  
    elif dataset_name=='products':
        data, texts= load_ogb_products()
    elif dataset_name=='citeseer':
        data, texts= load_citeseer()
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
    print('Number of classes:', len(torch.unique(data.y)))



    # ============================================
    # فاین‌تیون BERT (اگر مدل BiGTex باشه و finetune_lm=True)
    # ============================================
    pretrained_lm_model = None
    use_pretrained_lm = False
    finetune_lm=True
    if model_name == "BiGTex" and finetune_lm:
        print(f"\n{'#'*80}")
        print("# stage1: fineTuning the PLM")
        print(f"{'#'*80}\n")
        
        pretrained_lm_model = finetune_bert_with_soft_prompt(
            data=data,
            texts=texts,
            feature_dim=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            dataset_name=dataset_name,
            LM=LM,
            model_save_dir='finetuned_models',
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            device=device
        )
        
        # فریز کردن کامل مدل
        for param in pretrained_lm_model.parameters():
            param.requires_grad = False
        pretrained_lm_model.eval()
        
        use_pretrained_lm = True
        
        print(f"\n{'#'*80}")
        print("# Stage2: main training with Fine-tuned BERT ")
        print(f"{'#'*80}\n")
    
    # ============================================
    # آماده‌سازی DataLoader ها
    # ============================================
    train_loader, valid_loader, test_loader = prepare_loaders(data, batch_size)

    
    def _to_tensor(idx):
        if isinstance(idx, torch.Tensor):
            return idx
        return torch.tensor(idx, dtype=torch.long)
    
    train_idx = _to_tensor(data.train_idx).to(device)
   
    # ============================================
    # Training Loop 
    # ============================================
    train_acc = []
    val_acc = []
    test_acc = []

    for i in range(num_iterate):
        base_seed = 42
        current_seed = base_seed + i
        set_seed(current_seed)

        print(f"\nITERATION {i+1}/{num_iterate}")
        print("="*50)
      
        model = AdaptiveGraphTextModel(
                feature_dim=data.x.shape[1],
                text_embedding_dim=768,
                embedding_dim=embedding_dim,
                num_classes=len(torch.unique(data.y)),
                texts=texts,
                num_gcn_layers=num_layers,
                Lora=Lora,
                soft=soft,
                LM=LM,
                GNN=GNN,  # 'sage', 'gat', یا 'gcn'
                use_pretrained_lm=use_pretrained_lm,
                pretrained_lm_model=pretrained_lm_model, mode=mode
            ).to(device)
        # محاسبه پارامترها
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {model_name}")
        print(f"Mode_Fusion: {mode}")
        print(f"Language Model: {LM}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%\n")
      
        num_classes=len(torch.unique(data.y))

        train_, val_, test_ = train_model(
            model, train_idx, train_loader, valid_loader, test_loader, dataset_name, epochs,mode=mode,num_classes=num_classes,alpha=None)
       
        train_acc.append(train_)
        val_acc.append(val_)
        test_acc.append(test_)
        
#--------------------------------------------
#-------------C&S
    model1 = AdaptiveGraphTextModel(
                feature_dim=data.x.shape[1],
                text_embedding_dim=768,
                embedding_dim=embedding_dim,
                num_classes=len(torch.unique(data.y)),
                texts=texts,
                num_gcn_layers=num_layers,
                Lora=Lora,
                soft=soft,
                LM=LM,
                GNN=GNN,  # 'sage', 'gat', یا 'gcn'
                use_pretrained_lm=use_pretrained_lm,
                pretrained_lm_model=pretrained_lm_model, mode=mode
            ).to(device)
    model_name=dataset_name+"_model.pt"
    model1.load_state_dict(torch.load(model_name, map_location=device))
    model1 = model1.to(device)
    #-----------------------------
    #----------C&S

    def get_predictions_fast(model, data, device, batch_size=1024):
        """
        دریافت predictions با batch inference
        """
        from torch_geometric.loader import NeighborLoader
        
        model.eval()
        num_nodes = data.x.size(0)
        num_classes = len(torch.unique(data.y))
        
        # ساخت loader
        loader = NeighborLoader(
            data,
            num_neighbors=[-1] * 2,
            batch_size=batch_size,
            shuffle=False
        )
        
        # ذخیره predictions
        all_preds = torch.zeros(num_nodes, num_classes, device=device)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Getting predictions"):
                batch = batch.to(device)
                
                outputs,inal_hyb,final_plm,_, _ = model(
                    batch.x, 
                    batch.edge_index, 
                    batch.n_id, 
                    batch.x, 
                    batch.batch_size
                )
                
                # فقط seed nodes
                seed_outputs = outputs[:batch.batch_size]
                seed_ids = batch.n_id[:batch.batch_size]
                
                probs = F.softmax(seed_outputs, dim=-1)
                all_preds[seed_ids] = probs
        
        return all_preds


    def label_propagation_fast(y_soft, edge_index, num_nodes, mask, alpha=0.5, num_iters=50):
        """
        Label Propagation سریع با استفاده از sparse matrix operations
        """
        from torch_sparse import SparseTensor
        
        device = y_soft.device
        row, col = edge_index
        
        # ساخت normalized adjacency matrix
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # SparseTensor برای عملیات سریع‌تر
        adj = SparseTensor(
            row=row, col=col,
            value=deg_inv_sqrt[row] * deg_inv_sqrt[col],
            sparse_sizes=(num_nodes, num_nodes)
        ).to(device)
        
        # ذخیره مقادیر اولیه train nodes
        y_init = y_soft.clone()
        y_prop = y_soft.clone()
        
        for _ in range(num_iters):
            # Message passing سریع
            y_new = adj @ y_prop
            
            # ترکیب
            y_prop = alpha * y_new + (1 - alpha) * y_soft
            
            # ثابت نگه داشتن train nodes
            y_prop[mask] = y_init[mask]
        
        return y_prop

    def correct_and_smooth(
        model, 
        data, 
        device,
        correction_alpha=0.8,
        correction_iters=50,
        smoothing_alpha=0.8,
        smoothing_iters=50
    ):
        """
        نسخه بهینه Correct & Smooth
        """
        from ogb.nodeproppred import Evaluator
        
        print("\n" + "="*60)
        print("CORRECT & SMOOTH")
        print("="*60)
        
        num_nodes = data.x.size(0)
        num_classes = len(torch.unique(data.y))
        
        # ماسک train
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[data.train_idx] = True
        
        # =========================================
        # مرحله 1: دریافت predictions
        # =========================================
        print("\n[1/3] Getting predictions...")
        y_soft = get_predictions_fast(model, data, device)
        y_pred_base = y_soft.argmax(dim=-1)
        
        # =========================================
        # مرحله 2: CORRECT
        # =========================================
        print("[2/3] Correcting errors...")
        
        train_labels = data.y[train_mask].squeeze()
        train_onehot = F.one_hot(train_labels, num_classes).float()
        
        # محاسبه error
        errors = torch.zeros(num_nodes, num_classes, device=device)
        errors[train_mask] = train_onehot - y_soft[train_mask]
        
        # پخش errors
        errors_prop = label_propagation_fast(
            errors,
            data.edge_index,
            num_nodes,
            train_mask,
            alpha=correction_alpha,
            num_iters=correction_iters
        )
        
        # اصلاح
        y_corrected = y_soft + errors_prop
        y_corrected = y_corrected.clamp(min=0)
        y_corrected = y_corrected / y_corrected.sum(dim=1, keepdim=True)
        
        # =========================================
        # مرحله 3: SMOOTH
        # =========================================
        print("[3/3] Smoothing predictions...")
        
        # شروع با برچسب‌های train
        y_smooth = torch.zeros(num_nodes, num_classes, device=device)
        y_smooth[train_mask] = train_onehot
        
        # پخش
        y_smooth = label_propagation_fast(
            y_smooth,
            data.edge_index,
            num_nodes,
            train_mask,
            alpha=smoothing_alpha,
            num_iters=smoothing_iters
        )
        
        # ترکیب نهایی
        y_final = y_corrected + y_smooth
        y_final = y_final / y_final.sum(dim=1, keepdim=True)
        y_pred_final = y_final.argmax(dim=-1)
        
        # =========================================
        # ارزیابی
        # =========================================
        evaluator = Evaluator(name='ogbn-arxiv')
        
        test_idx = data.test_idx
        
        y_true = data.y[test_idx].cpu()
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        
        base_acc = evaluator.eval({
            'y_true': y_true,
            'y_pred': y_pred_base[test_idx].cpu().unsqueeze(1)
        })['acc']
        
        cs_acc = evaluator.eval({
            'y_true': y_true,
            'y_pred': y_pred_final[test_idx].cpu().unsqueeze(1)
        })['acc']
        
        improvement = cs_acc - base_acc
        
        print(f"\nTEST Results:")
        print(f"  Base:        {base_acc:.4f}")
        print(f"  After C&S:   {cs_acc:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({100*improvement/base_acc:+.2f}%)")
        print("="*60 + "\n")
        
        return y_pred_final, {
            'base_acc': base_acc,
            'cs_acc': cs_acc,
            'improvement': improvement
        }


#--------------------------------------------
#-------------C&S

       
    

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Mean Train Acc: {statistics.mean(train_acc):.4f} ± {statistics.stdev(train_acc):.4f}")
    print(f"Mean Val Acc: {statistics.mean(val_acc):.4f} ± {statistics.stdev(val_acc):.4f}")
    print(f"Mean Test Acc: {statistics.mean(test_acc):.4f} ± {statistics.stdev(test_acc):.4f}")
    #------------------------------
    #------------------C&S
    y_pred, results = correct_and_smooth(
            model1, data, device,
            correction_alpha=0.8,
            correction_iters=50,
            smoothing_alpha=0.87, 
            smoothing_iters=50
        )
    SaveEmbeddings(model,train_loader,valid_loader,test_loader, dataset_name, model_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*80}\n")
    


if __name__ == "__main__":
    main()


