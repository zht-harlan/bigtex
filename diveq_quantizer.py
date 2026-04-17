import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLevelDiVeQ(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size=128,
        quantizer_dim=None,
        use_straight_through=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.quantizer_dim = quantizer_dim or input_dim
        self.use_straight_through = use_straight_through

        self.input_projection = None
        self.output_projection = None
        if self.quantizer_dim != input_dim:
            self.input_projection = nn.Linear(input_dim, self.quantizer_dim)
            self.output_projection = nn.Linear(self.quantizer_dim, input_dim)

        self.codebook = nn.Parameter(torch.randn(codebook_size, self.quantizer_dim) * 0.02)

    def _project_in(self, ze):
        if self.input_projection is None:
            return ze
        return self.input_projection(ze)

    def _project_out(self, zq):
        if self.output_projection is None:
            return zq
        return self.output_projection(zq)

    def forward(self, ze):
        if ze.dim() != 2:
            raise ValueError(f"SingleLevelDiVeQ expects [B, D], got shape {tuple(ze.shape)}")

        ze_qspace = self._project_in(ze)
        ze_sq = (ze_qspace ** 2).sum(dim=1, keepdim=True)
        codebook_sq = (self.codebook ** 2).sum(dim=1).unsqueeze(0)
        distances = ze_sq + codebook_sq - 2 * ze_qspace @ self.codebook.t()

        indices = torch.argmin(distances, dim=1)
        quantized_qspace = F.embedding(indices, self.codebook)
        if self.use_straight_through:
            quantized_qspace = ze_qspace + (quantized_qspace - ze_qspace).detach()

        quantized = self._project_out(quantized_qspace)
        return {
            "indices": indices.unsqueeze(1),
            "quantized_token": quantized,
            "quantized_token_qspace": quantized_qspace,
            "quantized_stack": quantized_qspace.unsqueeze(1),
            "quantization_loss": ze.new_tensor(0.0),
            "commitment_loss": ze.new_tensor(0.0),
            "codebook_loss": ze.new_tensor(0.0),
        }
