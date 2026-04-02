import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ENCODER_ALIASES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "scibert": "allenai/scibert_scivocab_uncased",
    "deberta": "microsoft/deberta-v3-base",
}


def resolve_encoder_name(model_name):
    return ENCODER_ALIASES.get(model_name.lower(), model_name)


class OfflineTextEncoder:
    def __init__(
        self,
        model_name,
        pooling="cls",
        max_length=256,
        device=None,
        normalize=False,
    ):
        resolved_name = resolve_encoder_name(model_name)
        self.model_name = resolved_name
        self.pooling = pooling
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        self.model = AutoModel.from_pretrained(resolved_name).to(self.device)
        self.model.eval()

    def _pool_hidden_states(self, outputs, attention_mask):
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    def encode_texts(self, texts, batch_size=32, show_progress=True):
        all_embeddings = []
        iterable = range(0, len(texts), batch_size)
        if show_progress:
            iterable = tqdm(iterable, desc="Encoding texts")

        with torch.no_grad():
            for start in iterable:
                batch_texts = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                attention_mask = inputs["attention_mask"]
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                pooled = self._pool_hidden_states(outputs, attention_mask.to(self.device))
                all_embeddings.append(pooled.cpu())

        return torch.cat(all_embeddings, dim=0)
