import os
import torch
import torch.nn as nn

try:
    from .utils import PositionalEncoding
except ImportError:
    from utils import PositionalEncoding


# ============================================================
# Perceiver Lite Temporal — with residual gates
#
# Same architecture as TemporalPerceiver_concat but each
# residual connection is replaced by a highway-style gate:
#
#   gate  = sigmoid(W_g * pre_norm_input)   [B, T, d_model]
#   output = gate * sublayer_output + input
#
# This lets the model learn per-token, per-dimension how much
# of the new information should flow through versus keeping
# the current state.
# ============================================================

class PerceiverLiteTemporalGated(nn.Module):

    def __init__(
        self,
        d_in=1024,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_ff=1024,
        dropout=0.2,
        n_classes=3,
        K=8,
        mlp_hidden=256,
        mult=False,
        pe_type="sinusoidal",   # "sinusoidal" | "learned" | "none"
        pe_max_len=512,
        use_gate=True,          # highway-style gating on residuals
        readout="keyframe",     # "keyframe" | "mean" | "cls"
    ):
        super().__init__()

        self.mult = mult
        self.pe_type = pe_type
        self.use_gate = use_gate
        self.readout = readout

        # Projection EVA02 -> model dim
        self.in_proj = (
            nn.Identity()
            if d_in == d_model
            else nn.Linear(d_in, d_model)
        )

        self.in_norm = nn.LayerNorm(d_model)

        # Positional encoding
        if pe_type == "sinusoidal":
            self.pos_enc = PositionalEncoding(d_model=d_model, max_len=pe_max_len)
        elif pe_type == "learned":
            self.pos_enc = nn.Embedding(pe_max_len, d_model)
        else:  # "none"
            self.pos_enc = None

        # Latents
        self.latents = nn.Parameter(torch.randn(K, d_model) * 0.02)

        # CLS token: only used when readout == "cls"
        if readout == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            block = nn.ModuleDict({
                "ln1":   nn.LayerNorm(d_model),
                "cross": nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                "ln2":   nn.LayerNorm(d_model),
                "self":  nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                "ln3":   nn.LayerNorm(d_model),
                "ff":    nn.Sequential(
                    nn.Linear(d_model, dim_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_ff, d_model),
                    nn.Dropout(dropout)
                ),
            })
            if use_gate:
                block["gate_cross"] = nn.Linear(d_model, d_model, bias=True)
                block["gate_self"]  = nn.Linear(d_model, d_model, bias=True)
                block["gate_ff"]    = nn.Linear(d_model, d_model, bias=True)
            self.layers.append(block)

        self.out_norm = nn.LayerNorm(d_model)

        out_dim = 1 if mult else n_classes

        def make_head():
            if mlp_hidden > 0:
                return nn.Sequential(
                    nn.Linear(d_model, mlp_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden, out_dim),
                )
            return nn.Linear(d_model, out_dim)

        if mult:
            self.mult_head = nn.ModuleList([make_head() for _ in range(n_classes)])
            self.head = None
        else:
            self.head = make_head()
            self.mult_head = None

        if use_gate:
            for block in self.layers:
                nn.init.constant_(block["gate_cross"].bias, 0.0)
                nn.init.constant_(block["gate_self"].bias,  0.0)
                nn.init.constant_(block["gate_ff"].bias,    0.0)


    def _apply_pe(self, h, positions):
        """Add positional encoding to h [B, T, d_model]."""
        if self.pe_type == "sinusoidal":
            if positions is not None:
                pe_vals = self.pos_enc.pe[0][positions].to(h.dtype)  # [B, T, d_model]
                h = h + self.pos_enc.dropout(pe_vals)
            else:
                h = self.pos_enc(h)  # sequential fallback [0, 1, ..., T-1]
        elif self.pe_type == "learned":
            if positions is not None:
                h = h + self.pos_enc(positions).to(h.dtype)
            else:
                T = h.size(1)
                idx = torch.arange(T, device=h.device).unsqueeze(0)
                h = h + self.pos_enc(idx).to(h.dtype)
        # "none": no-op
        return h


    def forward(self, x, key_index, positions=None, return_attn: bool = False):
        """
        Parameters
        ----------
        return_attn : bool
            When True, also return cross-attention maps from every Perceiver
            layer as a tensor of shape [num_layers, B, K+1, T] (averaged over
            attention heads).  Useful for frame-importance visualisation.
            Caller receives ``(logits, attn_maps)`` instead of just ``logits``.
        """
        B, _, _ = x.shape
        device = x.device
        bidx = torch.arange(B, device=device)

        # Project + norm
        h = self.in_proj(x)
        h = self.in_norm(h)

        # PE applied before extracting key frame so position is encoded
        h = self._apply_pe(h, positions)

        # Key frame token extracted after PE
        key_feat = h[bidx, key_index]  # [B, d_model]

        # Build latent sequence [B, K+1, d_model]:
        #   "keyframe": [keyframe | lat_1 .. lat_K]  — readout from position 0
        #   "mean":     [keyframe | lat_1 .. lat_K]  — readout = mean of positions 1..K
        #   "cls":      [CLS      | lat_1 .. lat_K]  — readout from position 0 (CLS)
        lat_K = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, d_model]
        if self.readout == "cls":
            first_tok = self.cls_token.expand(B, -1, -1).contiguous()     # [B, 1, d_model]
        else:
            first_tok = key_feat.unsqueeze(1)                              # [B, 1, d_model]
        lat = torch.cat([first_tok, lat_K], dim=1)                        # [B, K+1, d_model]

        attn_maps = []  # [num_layers] × [B, K+1, T] — only filled when return_attn=True

        # Perceiver blocks
        for block in self.layers:
            # --- cross-attention ---
            q = block["ln1"](lat)
            if return_attn:
                cross, attn_w = block["cross"](q, h, h, need_weights=True, average_attn_weights=True)
                attn_maps.append(attn_w)  # [B, K+1, T]
            else:
                cross, _ = block["cross"](q, h, h, need_weights=False)
            if self.use_gate:
                gate = torch.sigmoid(block["gate_cross"](q))
                lat = gate * cross + lat
            else:
                lat = lat + cross

            # --- self-attention ---
            q2 = block["ln2"](lat)
            self_out, _ = block["self"](q2, q2, q2, need_weights=False)
            if self.use_gate:
                gate2 = torch.sigmoid(block["gate_self"](q2))
                lat = gate2 * self_out + lat
            else:
                lat = lat + self_out

            # --- feed-forward ---
            q3 = block["ln3"](lat)
            ff_out = block["ff"](q3)
            if self.use_gate:
                gate3 = torch.sigmoid(block["gate_ff"](q3))
                lat = gate3 * ff_out + lat
            else:
                lat = lat + ff_out

        # Readout
        if self.readout == "mean":
            # Mean of K latent tokens (positions 1..K); keyframe at position 0 provided context
            pooled = self.out_norm(lat[:, 1:].mean(dim=1))
        else:
            # "keyframe" or "cls": dedicated first token carries the readout
            pooled = self.out_norm(lat[:, 0])

        if self.mult:
            logits = torch.cat([head(pooled) for head in self.mult_head], dim=1)
        else:
            logits = self.head(pooled)

        if return_attn:
            return logits, torch.stack(attn_maps, dim=0)  # [num_layers, B, K+1, T]
        return logits


# ============================================================
# Helpers
# ============================================================

def _find_checkpoint_file(path: str) -> str:
    """Return the model weight file from a path or HF checkpoint directory."""
    if os.path.isfile(path):
        return path
    for fname in ("model.safetensors", "pytorch_model.bin"):
        candidate = os.path.join(path, fname)
        if os.path.isfile(candidate):
            return candidate
    # HF checkpoint-XXX subdirectory layout — pick the latest
    try:
        subdirs = sorted(
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.startswith("checkpoint-")
        )
    except FileNotFoundError:
        subdirs = []
    for subdir in reversed(subdirs):
        for fname in ("model.safetensors", "pytorch_model.bin"):
            candidate = os.path.join(path, subdir, fname)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"No checkpoint weight file found under: {path}")
