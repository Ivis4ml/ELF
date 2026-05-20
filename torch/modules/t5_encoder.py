"""Frozen T5 encoder, loaded via HuggingFace transformers.

The JAX repo reuses a custom pickled JAX/Flax T5 encoder. We do not attempt
to mirror that here — we load `google-t5/t5-small` (or whatever the config
points at) and freeze it. Numerically the encoder outputs match HuggingFace's
PyTorch implementation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import T5EncoderModel


class FrozenT5Encoder(nn.Module):
    """Wrap T5EncoderModel so that:
       - parameters do not require grad (frozen),
       - it always runs in eval() / no_grad,
       - it accepts an (B, L, L) self-attention mask in addition to (B, L).
    """

    def __init__(self, model_name: str = "google-t5/t5-small", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.d_model: int = self.model.config.d_model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """T5EncoderModel.forward accepts (B, L) keep masks directly, but recent
        transformers releases break on (B, L, L) 3D keep masks (they get reshaped
        to 5D inside masking_utils and the expand call fails). Workaround: turn the
        3D keep mask into a 4D (B, 1, L, L) additive mask, which the encoder passes
        straight through to scaled_dot_product_attention.
        """
        if attention_mask is not None and attention_mask.dim() == 3:
            keep = attention_mask.to(self.model.dtype)
            additive = (1.0 - keep) * torch.finfo(self.model.dtype).min
            attention_mask = additive.unsqueeze(1)  # (B, 1, L, L)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def get_encoder(model_name: str, dtype: torch.dtype = torch.float32) -> Tuple[FrozenT5Encoder, int]:
    """Match the JAX `get_encoder` signature loosely: returns the encoder and d_model."""
    encoder = FrozenT5Encoder(model_name, dtype=dtype)
    return encoder, encoder.d_model
