import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from diveq_quantizer import SingleLevelDiVeQ
from fusion_models import infer_lora_targets, resolve_backbone_name
from purified_graph_models import PurifiedGraphEncoder


class DiVeQGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        gnn_type="sage",
        num_layers=2,
        dropout=0.2,
        codebook_size=128,
        quantizer_dim=None,
        debug_shapes=True,
    ):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.shape_debug_printed = False

        self.encoder = PurifiedGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=True,
        )
        self.quantizer = SingleLevelDiVeQ(
            input_dim=hidden_dim,
            codebook_size=codebook_size,
            quantizer_dim=quantizer_dim,
            use_straight_through=True,
        )

    def forward(self, x, edge_index):
        ze = self.encoder.encode(x, edge_index)
        quantizer_outputs = self.quantizer(ze)
        zq = quantizer_outputs["quantized_token"]

        if self.debug_shapes and not self.shape_debug_printed:
            print(f"DiVeQ Ze shape: {tuple(ze.shape)}")
            print(f"DiVeQ code indices shape: {tuple(quantizer_outputs['indices'].shape)}")
            print(f"DiVeQ token shape: {tuple(zq.shape)}")
            self.shape_debug_printed = True

        return {
            "ze": ze,
            "zq": zq,
            "code_indices": quantizer_outputs["indices"],
            "quantized_stack": quantizer_outputs["quantized_stack"],
            "quantization_loss": quantizer_outputs["quantization_loss"],
            "commitment_loss": quantizer_outputs["commitment_loss"],
            "codebook_loss": quantizer_outputs["codebook_loss"],
        }


class CrossModalFusionDiVeQPLM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        gnn_type="sage",
        num_layers=2,
        graph_dropout=0.2,
        codebook_size=128,
        quantizer_dim=None,
        backbone_name="scibert",
        max_text_length=256,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        freeze_plm_embeddings=False,
        enable_vq_aux_head=False,
        debug_shapes=True,
    ):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.shape_debug_printed = False
        self.max_text_length = max_text_length

        resolved_backbone = resolve_backbone_name(backbone_name)
        self.backbone_name = resolved_backbone
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token

        self.plm = AutoModel.from_pretrained(resolved_backbone)
        self.plm_hidden_size = self.plm.config.hidden_size

        if freeze_plm_embeddings:
            for param in self.plm.get_input_embeddings().parameters():
                param.requires_grad = False

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=infer_lora_targets(resolved_backbone),
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.plm = get_peft_model(self.plm, lora_config)

        self.graph_encoder = DiVeQGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=graph_dropout,
            codebook_size=codebook_size,
            quantizer_dim=quantizer_dim,
            debug_shapes=debug_shapes,
        )

        self.struct_token_projection = nn.Linear(hidden_dim, self.plm_hidden_size)
        self.sep_projection = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.plm_hidden_size, self.plm_hidden_size),
            nn.ReLU(),
            nn.Dropout(graph_dropout),
            nn.Linear(self.plm_hidden_size, num_classes),
        )
        self.vq_aux_classifier = None
        if enable_vq_aux_head:
            self.vq_aux_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(graph_dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def _get_sep_embedding(self, batch_size, device):
        sep_id = self.tokenizer.sep_token_id
        if sep_id is None:
            sep_id = self.tokenizer.eos_token_id
        if sep_id is None:
            sep_id = self.tokenizer.cls_token_id
        if sep_id is None:
            raise ValueError("Tokenizer must define sep/eos/cls token for cross-modal separator.")

        sep_tokens = torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device)
        sep_embedding = self.plm.get_input_embeddings()(sep_tokens)
        return self.sep_projection(sep_embedding)

    def _tokenize_texts(self, texts, device):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tokens.items()}

    def forward(self, x, edge_index, texts, batch_size=None):
        graph_outputs = self.graph_encoder(x, edge_index)
        if batch_size is None:
            batch_size = x.size(0)

        code_indices = graph_outputs["code_indices"][:batch_size]
        ze = graph_outputs["ze"][:batch_size]
        zq = graph_outputs["zq"][:batch_size]

        device = x.device
        tokens = self._tokenize_texts(texts, device)
        text_embeddings = self.plm.get_input_embeddings()(tokens["input_ids"])

        struct_token = self.struct_token_projection(zq).unsqueeze(1)
        sep_embedding = self._get_sep_embedding(batch_size, device)

        fused_embeddings = torch.cat([struct_token, sep_embedding, text_embeddings], dim=1)
        struct_mask = torch.ones((batch_size, 1), dtype=tokens["attention_mask"].dtype, device=device)
        sep_mask = torch.ones((batch_size, 1), dtype=tokens["attention_mask"].dtype, device=device)
        attention_mask = torch.cat([struct_mask, sep_mask, tokens["attention_mask"]], dim=1)

        outputs = self.plm(inputs_embeds=fused_embeddings, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)

        if self.debug_shapes and not self.shape_debug_printed:
            print(f"DiVeQ struct token shape: {tuple(struct_token.shape)}")
            print(f"Text token embedding shape: {tuple(text_embeddings.shape)}")
            print(f"Fused embedding shape: {tuple(fused_embeddings.shape)}")
            print(f"Attention mask shape: {tuple(attention_mask.shape)}")
            self.shape_debug_printed = True

        vq_aux_logits = None
        if self.vq_aux_classifier is not None:
            vq_aux_logits = self.vq_aux_classifier(zq)

        return {
            "logits": logits,
            "attention_mask": attention_mask,
            "fused_embeddings": fused_embeddings,
            "ze": ze,
            "zq": zq,
            "code_indices": code_indices,
            "quantized_stack": graph_outputs["quantized_stack"][:batch_size],
            "vq_aux_logits": vq_aux_logits,
            "quantization_loss": graph_outputs["quantization_loss"],
            "commitment_loss": graph_outputs["commitment_loss"],
            "codebook_loss": graph_outputs["codebook_loss"],
        }
