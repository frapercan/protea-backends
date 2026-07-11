"""ProtST text-aligned embedding backend (ProtST-ESM1b).

Implements :class:`protea_contracts.EmbeddingBackend` for the ProtST
protein-text contrastive model (``mila-intel/ProtST-esm1b``). ProtST
projects an ESM-1b protein encoder into a joint protein/text embedding
space; the value of this backend is that projected, text-aligned
protein vector (the ``protein_feature`` head), which carries an
orthogonal signal to the raw PLM backbones (ESM / T5 / Ankh / ESM-C).

Port of the validated offline extraction recipe in
``storage/text_scorer/extract_protst.py``:

* Model:      ``mila-intel/ProtST-esm1b`` via
  ``AutoModel.from_pretrained(..., trust_remote_code=True)``. The
  checkpoint ships custom modelling code, so ``trust_remote_code`` is
  required (a deliberate, reviewable deviation from the HF-native
  backends).
* Tokenizer:  ``facebook/esm1b_t33_650M_UR50S``. The ProtST checkpoint
  reuses the base ESM-1b vocabulary and does not ship a tokenizer of
  its own, so the tokenizer identifier is fixed here rather than taken
  from ``EmbeddingConfig.model_name``.
* Forward:    ``model.protein_model(input_ids, attention_mask)`` and
  read ``out.protein_feature`` (the 512-d text-aligned projection).
* Precision:  fp32 (no fp16 cast). The reference extraction runs the
  projection head in full precision for numerical stability; this
  backend matches it rather than the fp16 path the raw-PLM backends use.

Three deviations from the raw-PLM backends, all inherent to a
whole-protein projection head (they mirror how the ESM-C plugin
deviates by not having a tokenizer):

1. **Single fixed vector.** ``protein_feature`` is already a pooled,
   whole-protein 512-d representation. The residue-level knobs
   (``layer_indices``, ``layer_agg``, ``pooling``, ``chunk_*``,
   ``normalize_residues``) do not apply; :meth:`embed_chunks` returns
   exactly one full-sequence :class:`ChunkEmbedding` per sequence and
   honours only ``config.normalize`` (L2).
2. **No per-residue path.** :meth:`embed_batch_per_residue` is left at
   the contract default (raises ``NotImplementedError``): there is no
   meaningful text-aligned per-residue projection to return.
3. **fp32 forward** (see above).

Heavy ML deps (``torch``, ``transformers``) are imported lazily inside
the methods so this module stays cheap to import for plugin discovery.
Install the runtime stack with ``pip install protea-backends[protst]``.

Example::

    from protea_backends.protst import plugin

    model, tok = plugin.load_model(
        "mila-intel/ProtST-esm1b", "cpu", emit=lambda *a, **k: None,
    )
    embeddings = plugin.embed_batch(
        model, tok, ["MSEQ"], emit=lambda *a, **k: None,
    )
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from protea_contracts import EmbeddingBackend

from protea_backends._chunk_helpers import ChunkEmbedding

#: ProtST-esm1b reuses the base ESM-1b vocabulary; the checkpoint ships
#: no tokenizer of its own, so the tokenizer id is fixed rather than
#: derived from ``EmbeddingConfig.model_name``.
_ESM1B_TOKENIZER = "facebook/esm1b_t33_650M_UR50S"

#: ESM-1b positional-embedding ceiling: 1024 tokens = 1022 residues plus
#: the BOS and EOS special tokens.
_MAX_RESIDUES = 1022

#: Dimensionality of the text-aligned ``protein_feature`` projection.
PROTEIN_FEATURE_DIM = 512


class ProtstBackend(EmbeddingBackend):
    """ProtST-ESM1b text-aligned backend (HuggingFace + ``trust_remote_code``)."""

    name = "protst"

    def load_model(
        self,
        model_name: str,
        device: str,
        emit: Any,
    ) -> tuple[Any, Any]:
        """Load the ProtST checkpoint + ESM-1b tokenizer, move to ``device``.

        ``model_name`` is the ProtST HuggingFace identifier (e.g.
        ``"mila-intel/ProtST-esm1b"``). The tokenizer is always
        ESM-1b's (:data:`_ESM1B_TOKENIZER`). The model is loaded with
        ``trust_remote_code=True`` (required by the checkpoint) and kept
        in fp32 to match the validated extraction recipe. Returns
        ``(model, tokenizer)`` in eval mode on ``device``.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        emit("backend.protst.load_start", None, {"model_name": model_name}, "info")
        _ = torch.device(device)  # validate the device string early
        tokenizer = AutoTokenizer.from_pretrained(_ESM1B_TOKENIZER)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        model.to(device)
        emit("backend.protst.load_done", None, {"model_name": model_name}, "info")
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
        """Return the ``(B, 512)`` text-aligned ``protein_feature`` matrix.

        ``layers`` / ``layer_agg`` / ``pooling`` are accepted for
        contract parity and ignored: ProtST emits a single fixed
        whole-protein projection, not a residue-level tensor to pool.
        Returns ``np.float16`` to match the other backends' storage dtype.
        """
        del layers, layer_agg, pooling
        if not sequences:
            return np.zeros((0, 0), dtype=np.float16)
        feats = _protein_features(model, tokenizer, sequences, _MAX_RESIDUES)
        emit("backend.protst.embed_done", None, {"n_sequences": len(sequences)}, "info")
        return cast("np.ndarray[Any, Any]", feats.astype(np.float16))

    def embed_chunks(
        self,
        model: Any,
        tokenizer: Any,
        sequences: list[str],
        config: Any,
        device: str,
    ) -> list[list[ChunkEmbedding]]:
        """Return one full-sequence ``ChunkEmbedding`` per sequence.

        This is the entry point ``protea-core`` dispatches to from
        ``compute_embeddings._dispatch_embed``. ProtST's
        ``protein_feature`` is already a pooled whole-protein vector, so
        the residue-level config knobs (``layer_indices``, ``pooling``,
        ``chunk_*``, ``normalize_residues``) do not apply; each sequence
        yields exactly one ``ChunkEmbedding`` covering the full sequence
        (``chunk_index_s=0``, ``chunk_index_e=None``). Only
        ``config.normalize`` is honoured (L2 over the 512-d vector).
        ``device`` is accepted for signature parity; the effective
        device is read from the model's parameters.
        """
        del device
        if not sequences:
            return []
        max_residues = min(int(getattr(config, "max_length", _MAX_RESIDUES)), _MAX_RESIDUES)
        feats = _protein_features(model, tokenizer, sequences, max_residues)
        if getattr(config, "normalize", False):
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(norms, 1e-12, None)
        feats = feats.astype(np.float32, copy=False)
        return [[ChunkEmbedding(0, None, feats[i])] for i in range(len(sequences))]


def _protein_features(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
    max_residues: int,
) -> np.ndarray[Any, Any]:
    """Tokenise a batch and return the ``(B, 512)`` fp32 ``protein_feature``.

    Truncates each sequence to ``max_residues`` residues (plus the BOS /
    EOS specials) before the padded-batch forward pass through
    ``model.protein_model``, matching ``extract_protst.py``. The
    ``protein_feature`` head is read defensively (attribute or mapping
    key) because ``trust_remote_code`` output types vary by revision.
    """
    import torch

    device_obj = next(model.parameters()).device
    enc = tokenizer(
        [s[:max_residues] for s in sequences],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_residues + 2,
    )
    enc = {k: v.to(device_obj) for k, v in enc.items()}
    with torch.no_grad():
        out = model.protein_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
    feat = out.protein_feature if hasattr(out, "protein_feature") else out["protein_feature"]
    result: np.ndarray[Any, Any] = feat.float().cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


#: Module-level plugin instance discovered via the ``protea.backends``
#: entry_points group.
plugin = ProtstBackend()
