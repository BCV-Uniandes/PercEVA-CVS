import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.TemporalPerceiver_concat_gate import PerceiverLiteTemporalGated


def _is_main_process() -> bool:
    """True on the global rank-0 process (or non-distributed runs)."""
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


class E2EPerceiverGated(nn.Module):
    def __init__(
        self,
        model_name: str = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = False,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 1024,
        dropout: float = 0.2,
        n_classes: int = 3,
        K: int = 8,
        mlp_hidden: int = 256,
        mult: bool = False,
        pe_type: str = "learned",
        pe_max_len: int = 2048,
        pos_weight=None,
    ):
        super().__init__()

        self.encoder = timm.create_model(model_name, pretrained=encoder_pretrained, num_classes=0)
        self.d_in = int(self.encoder.num_features)
        self.freeze_encoder = freeze_encoder
        self._n_unfrozen_blocks = 0 if freeze_encoder else len(self.encoder.blocks)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        self.perceiver = PerceiverLiteTemporalGated(
            d_in=self.d_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            n_classes=n_classes,
            K=K,
            mlp_hidden=mlp_hidden,
            mult=mult,
            pe_type=pe_type,
            pe_max_len=pe_max_len,
        )

        if pos_weight is None:
            pw = torch.ones(n_classes, dtype=torch.float32)
            self._pos_weight_active = False
        else:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
            self._pos_weight_active = True
        self.register_buffer("pos_weight", pw)
        self._pos_weight_forward_logged = False

        # ---- BCE loss-function validation (one-shot, init-time) ----
        # Test the EXACT BCE call the forward uses and compare against an
        # un-weighted reference. Asserts that pos_weight actually multiplies
        # the positive-class term and leaves the negative-class term untouched.
        with torch.no_grad():
            test_logits = torch.zeros(1, n_classes, dtype=torch.float32)  # sigmoid=0.5
            pos_labels = torch.ones(1, n_classes, dtype=torch.float32)
            neg_labels = torch.zeros(1, n_classes, dtype=torch.float32)

            l_no_pos = F.binary_cross_entropy_with_logits(test_logits, pos_labels, reduction="none")
            l_pw_pos = F.binary_cross_entropy_with_logits(
                test_logits, pos_labels, pos_weight=self.pos_weight, reduction="none"
            )
            l_no_neg = F.binary_cross_entropy_with_logits(test_logits, neg_labels, reduction="none")
            l_pw_neg = F.binary_cross_entropy_with_logits(
                test_logits, neg_labels, pos_weight=self.pos_weight, reduction="none"
            )
            ratio_pos = (l_pw_pos / l_no_pos.clamp(min=1e-12))[0].tolist()
            ratio_neg = (l_pw_neg / l_no_neg.clamp(min=1e-12))[0].tolist()
            expected_pos = self.pos_weight.tolist()

            pos_ok = all(abs(r - e) / max(abs(e), 1e-9) < 1e-3 for r, e in zip(ratio_pos, expected_pos))
            neg_ok = all(abs(r - 1.0) < 1e-3 for r in ratio_neg)
            self._pw_validation_passed = pos_ok and neg_ok

        if _is_main_process():
            marker = "ACTIVE" if self._pos_weight_active else "DISABLED (all-ones)"
            logging.info("BCE pos_weight %s — registered buffer = %s", marker, self.pos_weight.tolist())
            if not self._pw_validation_passed:
                logging.error("BCE pos_weight validation FAILED — loss function not applying weights correctly!")
                raise RuntimeError(
                    f"BCE pos_weight validation failed: pos_ratio={ratio_pos}, "
                    f"expected={expected_pos}, neg_ratio={ratio_neg}"
                )
            if not self._pos_weight_active:
                logging.warning(
                    "pos_weight is DISABLED — BCE runs without class re-weighting. "
                    "Pass --use_pos_weight=True or set use_pos_weight: true in YAML."
                )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._n_unfrozen_blocks == 0:
            self.encoder.eval()
        return self

    def unfreeze_encoder_last_n_blocks(self, n: int):
        """Freeze all encoder params, then unfreeze the last n transformer
        blocks + the final norm/fc_norm layers. n=0 -> full freeze;
        n>=total -> full unfreeze."""
        enc = self.encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        total = len(enc.blocks)
        n = max(0, min(int(n), total))
        if n == 0:
            self.freeze_encoder = True
            self._n_unfrozen_blocks = 0
            enc.eval()
            return
        for i in range(total - n, total):
            for p in enc.blocks[i].parameters():
                p.requires_grad_(True)
        for attr in ("norm", "fc_norm"):
            mod = getattr(enc, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad_(True)
        self.freeze_encoder = False
        self._n_unfrozen_blocks = n
        enc.train(self.training)
        logging.info("EVA02: blocks %d-%d + final norm unfrozen (of %d)", total - n, total - 1, total)

    def _extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = pixel_values.shape
        frames = pixel_values.reshape(B * T, C, H, W)
        total = len(self.encoder.blocks)
        n_unf = self._n_unfrozen_blocks
        if n_unf == 0:
            with torch.no_grad():
                feats = self.encoder(frames)
        elif n_unf >= total:
            feats = self.encoder(frames)
        else:
            feats = self._encode_split(frames)
        return feats.view(B, T, -1)

    def _encode_split(self, frames: torch.Tensor) -> torch.Tensor:
        """Two-phase forward: frozen prefix runs under no_grad (no activation
        graph stored), unfrozen tail runs with gradients. This is what saves
        the per-step activation memory."""
        enc = self.encoder
        cutoff = len(enc.blocks) - self._n_unfrozen_blocks
        with torch.no_grad():
            x = enc.patch_embed(frames)
            x, rope = enc._pos_embed(x)
            for blk in enc.blocks[:cutoff]:
                x = blk(x, rope=rope)
        x = x.detach()
        for blk in enc.blocks[cutoff:]:
            x = blk(x, rope=rope)
        if hasattr(enc, "norm") and enc.norm is not None:
            x = enc.norm(x)
        return enc.forward_head(x, pre_logits=True)

    def forward(self, pixel_values=None, key_index=None, labels=None, y=None, positions=None, **kwargs):
        if labels is None and y is not None:
            labels = y

        x = self._extract(pixel_values)                        # [B, T, D]
        logits = self.perceiver(x, key_index, positions)        # [B, n_classes]

        loss = None
        if labels is not None:
            labels = labels.to(dtype=logits.dtype)
            pw_used = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pw_used, reduction="none",
            ).mean()
            if not self._pos_weight_forward_logged and int(os.environ.get("RANK", "0")) == 0:
                self._pos_weight_forward_logged = True
                marker = "ACTIVE" if self._pos_weight_active else "DISABLED (all-ones)"
                logging.info(
                    "BCE first forward — pos_weight %s, used tensor=%s (dtype=%s, device=%s)",
                    marker, pw_used.detach().cpu().tolist(), pw_used.dtype, pw_used.device,
                )

        return {"loss": loss, "logits": logits}
