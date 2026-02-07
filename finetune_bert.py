# ================================================================
# فایل 1: finetune_bert.py
# فاین‌تیون BERT/GPT با LoRA و Soft Prompting
# ================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from transformers import DebertaV2Model, DebertaV2Tokenizer
from transformers import AutoModel, AutoTokenizer



from peft import LoraConfig, get_peft_model
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau



class NodeClassificationDataset(Dataset):
    """Dataset برای فاین‌تیون BERT روی تسک Node Classification"""
    def __init__(self, texts, features, labels, tokenizer, max_length=128):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'feature': torch.tensor(feature, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long).squeeze()

            # 'label': torch.tensor(label, dtype=torch.long)
        }


class BERTWithSoftPrompt(nn.Module):
    """مدل BERT/GPT با Soft Prompting و Classification Head"""
    def __init__(self, feature_dim, num_classes, text_embedding_dim=768, LM='BERT'):
        super(BERTWithSoftPrompt, self).__init__()
        
        self.LM = LM
        
        # انتخاب مدل زبانی
        if LM == "GPT":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model = GPT2Model.from_pretrained('gpt2')
        elif LM == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            self.lm_model = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        elif LM == "SCIBERT":
            self.tokenizer = BertTokenizer.from_pretrained(
                "allenai/scibert_scivocab_uncased", local_files_only=True
            )
            self.lm_model = BertModel.from_pretrained(
                "allenai/scibert_scivocab_uncased", local_files_only=True
            )
        elif LM == "DeBERTA":
                # self.text_model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
                # self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
                # self.lm_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
                # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
                self.lm_model = AutoModel.from_pretrained("models/deberta-v3-base" , local_files_only=True)
                self.tokenizer = AutoTokenizer.from_pretrained("models/deberta-v3-base", local_files_only=True)

        
        # فریز کردن پارامترهای اصلی
        for param in self.lm_model.parameters():
            param.requires_grad = False
        
        # اعمال LoRA
        # if LM in ["BERT", "DeBERTA", "SCIBERT"]:
        #     lora_config = LoraConfig(
        #         r=8,
        #         lora_alpha=32,
        #         target_modules=["attention.self.query", "attention.self.key", "attention.self.value"],
        #         lora_dropout=0.1,
        #         bias="none"
        #     )
        # elif LM == "GPT":
        #     lora_config = LoraConfig(
        #         r=8,
        #         lora_alpha=32,
        #         target_modules=["c_attn"],
        #         lora_dropout=0.1,
        #         bias="none"
        #     )
        if  LM in ["BERT", "SCIBERT"]:
            target_modules = ["attention.self.query", "attention.self.key", "attention.self.value"]
        elif  LM == "DeBERTA":
            target_modules = ["query_proj", "key_proj", "value_proj"]
        elif  LM == "GPT":
            target_modules = ["c_attn"]

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none"
        )
        
        self.lm_model = get_peft_model(self.lm_model, lora_config)
        
        # تبدیل feature به embedding برای soft prompt
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_embedding_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, features):
        # تبدیل feature به soft prompt
        soft_prompt = self.feature_transform(features)
        soft_prompt = soft_prompt.unsqueeze(1)
        
        # گرفتن embeddings متن
        input_embeddings = self.lm_model.get_input_embeddings()(input_ids)
        
        # Concatenate کردن soft prompt به ابتدای متن
        modified_embeddings = torch.cat([soft_prompt, input_embeddings], dim=1)
        
        # تنظیم attention mask
        batch_size = attention_mask.shape[0]
        new_token_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([new_token_mask, attention_mask], dim=1)
        
        # عبور از LM
        outputs = self.lm_model(inputs_embeds=modified_embeddings, attention_mask=extended_attention_mask)
        
        # استفاده از [CLS] token (اولین توکن بعد از soft prompt)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits


def finetune_bert_with_soft_prompt(
    data,
    texts, 
    feature_dim,
    num_classes,
    dataset_name,
    LM='BERT',
    model_save_dir='finetuned_models',
    epochs=5,
    batch_size=32,
    learning_rate=2e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
   
    
    # ساخت دایرکتوری
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'finetuned_{LM.lower()}_{dataset_name}.pt')
    
    # چک کردن اینکه مدل از قبل وجود داره یا نه
    if os.path.exists(model_save_path):
        print(f"\n{'='*80}")
        print(f"✅ مدل {LM} فاین‌تیون شده برای {dataset_name} از قبل وجود دارد")
        print(f"   Path: {model_save_path}")
        print(f"   Loading Fine-tuned model...")
        print(f"{'='*80}\n")
        
        model = BERTWithSoftPrompt(feature_dim, num_classes, LM=LM)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    
    # شروع فاین‌تیون
    print(f"\n{'='*80}")
    print(f"🚀 start fine-tuning {LM} with LoRA and Soft Prompting")
    print(f"   Dataset: {dataset_name}")
    print(f"{'='*80}\n")
    #-----------------------------validation function
    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features_batch = batch['feature'].to(device)
                labels_batch = batch['label'].to(device)

                logits = model(input_ids, attention_mask, features_batch)
                _, predicted = torch.max(logits, 1)

                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        acc = 100 * correct / total
        model.train()
        return acc

    # استخراج train data
    train_idx = data.train_idx
    if isinstance(train_idx, torch.Tensor):
        train_idx = train_idx.cpu().numpy()
    
    train_texts = [texts[i] for i in train_idx]
    train_features = data.x[train_idx].cpu().numpy()
    train_labels = data.y[train_idx].cpu().numpy()
        # ================= Validation & Test Data =================
    val_idx = data.valid_idx
    if isinstance(val_idx, torch.Tensor):
        val_idx = val_idx.cpu().numpy()

    test_idx = data.test_idx
    if isinstance(test_idx, torch.Tensor):
        test_idx = test_idx.cpu().numpy()

    val_texts = [texts[i] for i in val_idx]
    val_features = data.x[val_idx].cpu().numpy()
    val_labels = data.y[val_idx].cpu().numpy()

    test_texts = [texts[i] for i in test_idx]
    test_features = data.x[test_idx].cpu().numpy()
    test_labels = data.y[test_idx].cpu().numpy()

    
    print(f"📊 Information:")
    print(f"   - Number of train: {len(train_texts)}")
    print(f"   - Dim of  feature: {feature_dim}")
    print(f"   - num of classes: {num_classes}\n")
    
    # ساخت مدل
    model = BERTWithSoftPrompt(feature_dim, num_classes, LM=LM)
    model.to(device)
    
    # Dataset و DataLoader
    train_dataset = NodeClassificationDataset(
        texts=train_texts,
        features=train_features,
        labels=train_labels,
        tokenizer=model.tokenizer,
        max_length=128 if LM in ["BERT", "DeBERTA", "SCIBERT"] else 30
    )
    val_dataset = NodeClassificationDataset(
        texts=val_texts,
        features=val_features,
        labels=val_labels,
        tokenizer=model.tokenizer,
        max_length=128 if LM in ["BERT", "DeBERTA", "SCIBERT"] else 30
    )

    test_dataset = NodeClassificationDataset(
        texts=test_texts,
        features=test_features,
        labels=test_labels,
        tokenizer=model.tokenizer,
        max_length=128 if LM in ["BERT", "DeBERTA", "SCIBERT"] else 30
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer و Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

    criterion = nn.CrossEntropyLoss()
    
    # نمایش تعداد پارامترها
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Information of Model:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters (LoRA): {trainable_params:,}")
    print(f"   - Trainable percentage: {100 * trainable_params / total_params:.2f}%\n")
    
    # Training Loop
    model.train()
    best_acc = 0.0
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features_batch = batch['feature'].to(device)
            labels_batch = batch['label'].to(device)
            
            # Forward
            logits = model(input_ids, attention_mask, features_batch)
            loss = criterion(logits, labels_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            _, predicted = torch.max(logits, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
                # ================= Evaluation =================
        train_acc = accuracy
        val_acc = evaluate(model, val_loader)
        test_acc = evaluate(model, test_loader)

        print(
            f"   📊 Epoch {epoch+1}: "
            f"Train Acc = {train_acc:.2f}% | "
            f"Val Acc = {val_acc:.2f}% | "
            f"Test Acc = {test_acc:.2f}%"
        )

        # ذخیره بهترین مدل بر اساس validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"   ✅ Saved Best Model (Val Acc: {val_acc:.2f}%)")

        # print(f'   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        scheduler.step(val_acc)

        # ذخیره بهترین مدل
    #     if accuracy > best_acc:
    #         best_acc = accuracy
    #         torch.save(model.state_dict(), model_save_path)
    #         print(f'   ✅ Seved the best model (Acc: {accuracy:.2f}%)')
    
    # print(f"\n✅ finished fine-tuning {LM} !")
    # print(f"   Saved the model: {model_save_path}")
    # print(f"   The best Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}\n")
    
    return model

