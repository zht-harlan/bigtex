import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from residual_quantizer import ResidualQuantizer


class PurifiedGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        gnn_type="sage",
        num_layers=2,
        dropout=0.2,
        use_residual=True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(self._build_conv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_conv(self, in_dim, out_dim):
        if self.gnn_type == "sage":
            return SAGEConv(in_dim, out_dim)
        if self.gnn_type == "gcn":
            return GCNConv(in_dim, out_dim)
        if self.gnn_type == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        raise ValueError(f"Unsupported gnn_type: {self.gnn_type}")

    def encode(self, x, edge_index):
        h = self.input_projection(x)
        h = self.input_norm(h)

        for conv, norm in zip(self.gnn_layers, self.layer_norms):
            h_next = conv(h, edge_index)
            h_next = F.relu(h_next)
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            if self.use_residual and h_next.shape == h.shape:
                h = norm(h + h_next)
            else:
                h = norm(h_next)

        return h

    def forward(self, x, edge_index):
        ze = self.encode(x, edge_index)
        logits = self.classifier(ze)
        return logits, ze


class QuantizedPurifiedGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        gnn_type="sage",
        num_layers=2,
        dropout=0.2,
        use_residual=True,
        codebook_size=128,
        num_quantizers=3,
        quantizer_dim=None,
        use_straight_through=True,
        use_commitment_loss=True,
        commitment_weight=0.25,
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
            use_residual=use_residual,
        )
        self.quantizer = ResidualQuantizer(
            input_dim=hidden_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            quantizer_dim=quantizer_dim,
            use_straight_through=use_straight_through,
            use_commitment_loss=use_commitment_loss,
            commitment_weight=commitment_weight,
        )
        self.quantized_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_continuous(self, x, edge_index):
        return self.encoder.encode(x, edge_index)

    def forward(self, x, edge_index):
        ze = self.encode_continuous(x, edge_index)
        quantizer_outputs = self.quantizer(ze)
        zq = quantizer_outputs["quantized_sum"]
        logits = self.quantized_classifier(zq)

        if self.debug_shapes and not self.shape_debug_printed:
            print(f"Ze shape: {tuple(ze.shape)}")
            print(f"Code indices shape: {tuple(quantizer_outputs['indices'].shape)}")
            print(f"Quantized output shape: {tuple(zq.shape)}")
            self.shape_debug_printed = True

        return {
            "logits": logits,
            "ze": ze,
            "zq": zq,
            "code_indices": quantizer_outputs["indices"],
            "quantized_stack": quantizer_outputs["quantized_stack"],
            "residual_states": quantizer_outputs["residual_states"],
            "quantization_loss": quantizer_outputs["quantization_loss"],
            "commitment_loss": quantizer_outputs["commitment_loss"],
            "codebook_loss": quantizer_outputs["codebook_loss"],
        }
