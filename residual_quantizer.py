import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_quantizers=3,
        codebook_size=128,
        quantizer_dim=None,
        use_straight_through=True,
        use_commitment_loss=True,
        commitment_weight=0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.quantizer_dim = quantizer_dim or input_dim
        self.use_straight_through = use_straight_through
        self.use_commitment_loss = use_commitment_loss
        self.commitment_weight = commitment_weight

        self.input_projection = None
        self.output_projection = None
        if self.quantizer_dim != input_dim:
            self.input_projection = nn.Linear(input_dim, self.quantizer_dim)
            self.output_projection = nn.Linear(self.quantizer_dim, input_dim)

        self.codebooks = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, self.quantizer_dim) * 0.02
        )

    def _project_in(self, ze):
        if self.input_projection is None:
            return ze
        return self.input_projection(ze)

    def _project_out(self, quantized):
        if self.output_projection is None:
            return quantized
        return self.output_projection(quantized)

    def _quantize_once(self, residual, codebook):
        residual_sq = (residual ** 2).sum(dim=1, keepdim=True)
        codebook_sq = (codebook ** 2).sum(dim=1).unsqueeze(0)
        distances = residual_sq + codebook_sq - 2 * residual @ codebook.t()
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, codebook)
        return indices, quantized

    def forward(self, ze):
        if ze.dim() != 2:
            raise ValueError(f"ResidualQuantizer expects [B, D], got shape {tuple(ze.shape)}")

        ze_qspace = self._project_in(ze)
        residual = ze_qspace
        quantized_chunks = []
        indices_per_level = []
        residual_states = [residual]
        commitment_losses = []
        codebook_losses = []

        for level in range(self.num_quantizers):
            codebook = self.codebooks[level]
            indices, quantized_level = self._quantize_once(residual, codebook)
            indices_per_level.append(indices)
            quantized_chunks.append(quantized_level)

            if self.use_commitment_loss:
                commitment_losses.append(
                    F.mse_loss(residual, quantized_level.detach())
                )
                codebook_losses.append(
                    F.mse_loss(quantized_level, residual.detach())
                )

            residual = residual - quantized_level
            residual_states.append(residual)

        quantized_stack = torch.stack(quantized_chunks, dim=1)
        quantized_sum_qspace = quantized_stack.sum(dim=1)

        if self.use_straight_through:
            quantized_sum_qspace = ze_qspace + (quantized_sum_qspace - ze_qspace).detach()

        quantized_sum = self._project_out(quantized_sum_qspace)
        indices = torch.stack(indices_per_level, dim=1)

        if self.use_commitment_loss and commitment_losses:
            commitment_loss = torch.stack(commitment_losses).mean()
            codebook_loss = torch.stack(codebook_losses).mean()
            quantization_loss = self.commitment_weight * commitment_loss + codebook_loss
        else:
            commitment_loss = ze.new_tensor(0.0)
            codebook_loss = ze.new_tensor(0.0)
            quantization_loss = ze.new_tensor(0.0)

        return {
            "indices": indices,
            "quantized_stack": quantized_stack,
            "quantized_sum": quantized_sum,
            "quantized_sum_qspace": quantized_sum_qspace,
            "residual_states": residual_states,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "quantization_loss": quantization_loss,
        }
