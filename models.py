
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
from transformers import DebertaV2Model, DebertaV2Tokenizer
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from peft import get_peft_model, LoraConfig
from transformers import AutoModel, AutoTokenizer

class AdaptiveGraphTextModel(nn.Module):

    def __init__(self, feature_dim, text_embedding_dim, num_classes, texts, 
                 embedding_dim=128, num_gcn_layers=2, Lora=True, soft=True, LM='BERT', GNN='sage',
                 use_pretrained_lm=False, pretrained_lm_model=None,
                 confidence_threshold=0.6,mode="AE", degree_threshold_percentile=25,
                 use_adaptive=True):

        super(AdaptiveGraphTextModel, self).__init__()
        
        self.texts = texts
        self.soft = soft
        self.LM = LM
        self.GNN = GNN
        self.use_pretrained_lm = use_pretrained_lm
        self.use_adaptive = use_adaptive
        self.confidence_threshold = confidence_threshold
        self.degree_threshold_percentile = degree_threshold_percentile
        self.mode= mode


        self.graph_proj = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.text_proj  = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.tau = 0.5

        # تنظیم مدل زبانی
        if use_pretrained_lm and pretrained_lm_model is not None:
            print("   ✅ استفاده از مدل زبانی فاین‌تیون شده (فریز شده)")
            
            self.tokenizer = pretrained_lm_model.tokenizer
            self.text_model = pretrained_lm_model.lm_model
            self.pretrained_feature_transform = pretrained_lm_model.feature_transform
            
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_feature_transform.parameters():
                param.requires_grad = False


                
        else:
            print("   ⚠️ ساخت مدل زبانی جدید (با LoRA - end-to-end training)")
            
            if LM == "GPT":
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.text_model = GPT2Model.from_pretrained('gpt2')
            elif LM == "BERT":
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.text_model = BertModel.from_pretrained("bert-base-uncased")
            elif LM == "DeBERTA":
                self.text_model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
                self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
                # self.text_model = AutoModel.from_pretrained("models/deberta-v3-base" )
                # self.tokenizer = AutoTokenizer.from_pretrained("models/deberta-v3-base")
           

            elif LM == "SCIBERT":
                self.tokenizer = BertTokenizer.from_pretrained(
                    "allenai/scibert_scivocab_uncased"
                )
                self.text_model = BertModel.from_pretrained(
                    "allenai/scibert_scivocab_uncased"
                )

            
            
                
                
            for param in self.text_model.parameters():
                param.requires_grad = False
            if Lora and LM in ["BERT", "SCIBERT"]:
                target_modules = ["attention.self.query", "attention.self.key", "attention.self.value"]
            elif Lora and LM == "DeBERTA":
                target_modules = ["query_proj", "key_proj", "value_proj"]
            elif Lora and LM == "GPT":
                target_modules = ["c_attn"]

            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none"
            )
        
            self.text_model = get_peft_model(self.text_model, lora_config)
            
            self.pretrained_feature_transform = None
        
        # Feature transform برای گراف
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_embedding_dim)
        )
        
        # GNN layers
        assert num_gcn_layers >= 2, "برای adaptive mode حداقل 2 لایه GNN نیاز است"
        
        if self.GNN == 'sage':
            self.gcn_layers = nn.ModuleList([  
                SAGEConv(text_embedding_dim, text_embedding_dim)   
                for _ in range(num_gcn_layers)  
            ])
        elif self.GNN == 'gat':
            self.gcn_layers = nn.ModuleList([  
                GATConv(text_embedding_dim, text_embedding_dim)   
                for _ in range(num_gcn_layers)  
            ])
        elif self.GNN == 'gcn':
            self.gcn_layers = nn.ModuleList([  
                GCNConv(text_embedding_dim, text_embedding_dim)   
                for _ in range(num_gcn_layers)  
            ])

        
        self.proj1 = nn.Linear(text_embedding_dim, text_embedding_dim)
        

        self.norm_graph = nn.LayerNorm(text_embedding_dim)
        self.norm_text  = nn.LayerNorm(text_embedding_dim)
        self.norm_  = nn.LayerNorm(text_embedding_dim)
        if self.mode=="AE":
        # Autoencoder for graph-text fusion
            self.ae_encoder = nn.Sequential(
                nn.Linear(2 * text_embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, text_embedding_dim)   # bottleneck z
            )

            self.ae_decoder = nn.Sequential(
                nn.Linear(text_embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2 * text_embedding_dim)
            )
        elif self.mode == "MLP":
            self.proj2 = nn.Sequential(
            nn.Linear( 2* text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_embedding_dim)
            )
        elif self.mode == "cross":
            self.cross_attention = nn.MultiheadAttention(text_embedding_dim, 4)

        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(num_classes)
        ])

        self.MLP0 = nn.Sequential(
            nn.Linear(text_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, text_embedding_dim)
        )
        

    

    def compute_seed_mask(self, embeddings, edge_index, batch_size, layer_idx):
        device = embeddings.device
        seed_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
       
        return seed_mask, {}
    
    def apply_bert_to_seed_nodes(self, graph_embedding, seed_mask, seed_input_embeddings, edge_index, batch_size):
        device = graph_embedding.device
        
        if not seed_mask.any():
            return graph_embedding
        
        seed_embeddings = graph_embedding[:batch_size]
        seed_input_embeddings = seed_input_embeddings[seed_mask]

        seed_seed_embeddings = seed_embeddings[seed_mask]

        if self.soft:
            seed_graph_embedding = self.proj1(seed_seed_embeddings)
            seed_graph_embedding = seed_graph_embedding.unsqueeze(1)
            
            modified_embeddings = torch.cat([seed_graph_embedding, seed_input_embeddings], dim=1)

            num_seed = seed_input_embeddings.size(0)
            seq_len = seed_input_embeddings.size(1)
            
            attention_mask = torch.ones((num_seed, 1 + seq_len), 
                                        dtype=torch.long, device=device)
            
            outputs = self.text_model(inputs_embeds=modified_embeddings, 
                                    attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            text_embedding = hidden_states[:, 0, :]
            
            graph_emb_for_concat = seed_graph_embedding.squeeze(1).unsqueeze(0)
            text_emb_for_concat = text_embedding.unsqueeze(0)

            text_emb_for_concat = self.norm_text(text_emb_for_concat)
            graph_emb_for_concat = self.norm_graph(graph_emb_for_concat)
            if self.mode == "AE":
                combined_embedding = torch.cat([graph_emb_for_concat, text_emb_for_concat], dim=2)
                latent_z = self.ae_encoder(combined_embedding)  # [num_seed, d]
            elif self.mode == "MLP":
                combined_embedding = torch.cat([graph_emb_for_concat, text_emb_for_concat], dim=2)
                # combined_embedding= graph_emb_for_concat + (graph_emb_for_concat * text_emb_for_concat) *text_emb_for_concat
                latent_z = self.proj2(combined_embedding)
            if self.mode == "cross":
                latent_z, _ = self.cross_attention(graph_emb_for_concat,text_emb_for_concat,text_emb_for_concat)  
                
            
            refined_seed_embedding = latent_z
        else:
            outputs = self.text_model(inputs_embeds=seed_input_embeddings)
            text_embedding = outputs.last_hidden_state[:, 0, :]
            refined_seed_embedding = text_embedding
        
        refined_seed_embeddings = seed_embeddings.clone()
        refined_seed_embeddings[seed_mask] = refined_seed_embedding
        
        refined_embedding = graph_embedding.clone()
        refined_embedding[:batch_size] = refined_seed_embeddings
        
        return refined_embedding, text_emb_for_concat
    
    def apply_class_heads(self, z, heads):
            logits = [head(z) for head in heads]  # لیست [N,1]
            return torch.cat(logits, dim=1)       # [N, C]
    def forward(self, x, edge_index, n_id, feature_vec, batch_size):
      
   
        
        h=[]
        if self.use_pretrained_lm and self.pretrained_feature_transform is not None:
            transformed_feature = self.pretrained_feature_transform(feature_vec)
        else:
            transformed_feature = self.feature_transform(feature_vec)

        graph_embedding = transformed_feature
        edge_index = edge_index.long()
        
        seed_n_ids = n_id[:batch_size].cpu().numpy()
        seed_texts = [self.texts[i] for i in seed_n_ids]
        
        if self.LM in ["BERT", "DeBERTA", "SCIBERT"]:
            tokens = self.tokenizer(seed_texts, padding=True, truncation=True, 
                                max_length=128, return_tensors='pt')
        elif self.LM == "GPT":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokens = self.tokenizer(seed_texts, padding=True, truncation=True, 
                                max_length=30, return_tensors='pt')
        
        tokens = tokens.to(edge_index.device)
        seed_input_embeddings = self.text_model.get_input_embeddings()(tokens['input_ids'])
        
        first_gcn = self.gcn_layers[0]
        # h.append(torch.nn.functional.softmax(self.classifier(graph_embedding), dim=1))
        logits_layer = self.apply_class_heads(graph_embedding, self.class_heads)
        h.append(torch.softmax(logits_layer, dim=1))

        graph_embedding = first_gcn(graph_embedding, edge_index)
        graph_embedding= self.MLP0(graph_embedding)
        # h.append(torch.nn.functional.softmax(self.classifier(graph_embedding), dim=1))
        logits_layer = self.apply_class_heads(graph_embedding, self.class_heads)
        h.append(torch.softmax(logits_layer, dim=1))

 
        for layer_idx, gcn_layer in enumerate(self.gcn_layers[1:], start=1):
            if self.use_adaptive:
                seed_embeddings = graph_embedding[:batch_size]
                seed_mask, stats = self.compute_seed_mask(seed_embeddings, edge_index, batch_size, layer_idx)
            else:
                seed_mask = torch.ones(batch_size, dtype=torch.bool, device=graph_embedding.device)
            
            if seed_mask.any():
                graph_embedding, text_emb_seed_before_concat = self.apply_bert_to_seed_nodes(graph_embedding, seed_mask, seed_input_embeddings, edge_index, batch_size)
            
            # graph_embedding = gcn_layer(graph_embedding, edge_index)
            h_in = graph_embedding
            h_out = gcn_layer(h_in, edge_index)
            graph_embedding = h_out + h_in
            graph_embedding= self.norm_(graph_embedding)
            # h.append(torch.nn.functional.softmax(self.classifier(graph_embedding), dim=1))
            logits_layer = self.apply_class_heads(graph_embedding, self.class_heads)
            h.append(torch.softmax(logits_layer, dim=1))

        
        

        final_output = self.apply_class_heads(graph_embedding, self.class_heads)
        g_proj = F.normalize(self.graph_proj(graph_embedding[:batch_size]), dim=1)
        text_emb_seed_before_concat = text_emb_seed_before_concat.squeeze(0)
        t_proj = F.normalize(self.text_proj(text_emb_seed_before_concat), dim=1)


        
        return final_output, h, graph_embedding, graph_embedding[:batch_size], text_emb_seed_before_concat

 
