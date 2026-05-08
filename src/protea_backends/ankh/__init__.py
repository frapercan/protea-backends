"""Ankh embedding backend (Ankh-base, Ankh-large).

Ankh is a T5 architecture trained by ElnaggarLab for protein
sequences. It ships through HuggingFace ``T5EncoderModel`` but has
two quirks that distinguish it from generic ProtT5:

* **bfloat16 on CUDA**. Ankh was pre-trained on TPU in bfloat16; its
  LayerNorm overflows in FP16, every forward collapses to NaN. Use
  bfloat16 on CUDA (same VRAM footprint as FP16 but FP32 dynamic
  range) and FP32 on CPU. Verified on ``ElnaggarLab/ankh-base``
  2026-04-10.
* **is_split_into_words=True** for tokenisation. Ankh's SentencePiece
  tokeniser maps a literal space to ``<unk>``, so the space-joined
  path used by ProtT5/ProstT5 produces ~50% ``<unk>`` tokens. Pass
  the sequence as a list of characters with
  ``is_split_into_words=True`` so each residue becomes one word.

Heavy ML deps (torch, transformers) are imported lazily; install the
runtime stack with ``pip install protea-backends[ankh]`` (extras land
in F2A.5 of master plan v3).

Example::

    from protea_backends.ankh import plugin

    model, tok = plugin.load_model(
        "ElnaggarLab/ankh-base",
        "cpu",
        emit=lambda *a, **k: None,
    )
    embeddings = plugin.embed_batch(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend


class AnkhBackend(EmbeddingBackend):
    """Ankh family backend (HuggingFace T5EncoderModel + AutoTokenizer)."""

    name = "ankh"

    def load_model(
        self,
        model_name: str,
        device: str,
        emit: Any,
    ) -> tuple[Any, Any]:
        """Load an Ankh checkpoint + AutoTokenizer, move to ``device``.

        ``model_name`` is a HuggingFace identifier (e.g.
        ``"ElnaggarLab/ankh-base"`` or ``"ElnaggarLab/ankh-large"``).
        Returns ``(model, tokenizer)``. **bfloat16** on CUDA (FP16
        LayerNorm collapses to NaN), FP32 on CPU.

        AutoTokenizer is used (resolves to T5TokenizerFast) instead of
        hardcoding T5Tokenizer; some Ankh revisions changed the
        tokenizer class and AutoTokenizer rides those changes.
        """
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        emit("backend.ankh.load_start", None, {"model_name": model_name}, "info")
        device_obj = torch.device(device)
        # Ankh-specific: bfloat16 on CUDA, fp32 on CPU.
        dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=dtype,
        )
        model.eval()
        model.to(device)
        emit(
            "backend.ankh.load_done",
            None,
            {"model_name": model_name, "dtype": str(dtype)},
            "info",
        )
        return model, tokenizer

    def embed_batch(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        *,
        emit: Any,
        layers: list[int] | None = None,
        layer_agg: str = "mean",
        pooling: str = "mean",
    ) -> np.ndarray[Any, Any]:
        """Run inference on a batch and return mean-pooled embeddings.

        Tokenises each sequence as a list of characters with
        ``is_split_into_words=True`` so the SentencePiece tokeniser
        treats every residue as one word and never falls back to
        ``<unk>``. Strips trailing EOS to align with the other
        backends. Returns ``(batch_size, hidden_dim)`` float16 ndarray
        (output cast back to fp16 for storage parity even though the
        model runs in bfloat16).
        """
        import torch

        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)

        cleaned = [re.sub(r"[UZOB]", "X", s) for s in sequences]
        device_obj = next(model.parameters()).device

        # Ankh-specific: list-of-chars with is_split_into_words=True.
        inputs = tokenizer.batch_encode_plus(
            [list(c) for c in cleaned],
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        del outputs

        use_layers = layers if layers else [0]
        stack = torch.stack(
            [hidden_states[-(li + 1)] for li in use_layers], dim=0
        )
        if layer_agg == "mean":
            layer_pooled = stack.mean(dim=0)
        elif layer_agg == "sum":
            layer_pooled = stack.sum(dim=0)
        else:
            layer_pooled = stack.mean(dim=0)

        attn = inputs["attention_mask"]

        out: list[np.ndarray[Any, Any]] = []
        for i in range(layer_pooled.shape[0]):
            actual_len = int(attn[i].sum().item())
            # No CLS / no AA2fold; only strip trailing EOS.
            residues = layer_pooled[i, : actual_len - 1, :].float()
            if pooling in ("mean", "max"):
                pooled = (
                    residues.mean(dim=0)
                    if pooling == "mean"
                    else residues.max(dim=0).values
                )
            else:
                pooled = residues.mean(dim=0)
            out.append(pooled.cpu().numpy().astype(np.float16))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        emit(
            "backend.ankh.embed_done",
            None,
            {"n_sequences": len(sequences)},
            "info",
        )
        return cast("np.ndarray[Any, Any]", np.stack(out).astype(np.float16))


#: Module-level plugin instance discovered via the
#: ``protea.backends`` entry_points group.
plugin = AnkhBackend()
