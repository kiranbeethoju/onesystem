"""OneSystem: a standalone System 1 decision model.

OneSystem reads a document together with the labels that are legal for a
decision and scores every label in one forward pass. It is its own model: a
bidirectional encoder plus a label-scoring head, saved as plain safetensors.
It does not load or depend on GLiNER checkpoints.
"""

__version__ = "0.3.0"

MODEL_NAME = "OneSystem"
# Encoder used to initialise training. After training, OneSystem ships the full
# encoder weights under its own name and never downloads this checkpoint again.
# Long-context embedding backbone (8192). Prefer contrastive/embedding models
# over plain LMs — fine-tuning ModernBERT collapsed cosine geometry.
DEFAULT_ENCODER = "Alibaba-NLP/gte-base-en-v1.5"
ENCODER_MAX_POSITIONS = 8192
DATASET_ID = "fastino/fast-decisions"
RELEASE_REPO = "kiranbeethoju/onesystem"
RELEASE_TAG = "v0.3.0"

__all__ = [
    "DATASET_ID",
    "DEFAULT_ENCODER",
    "ENCODER_MAX_POSITIONS",
    "MODEL_NAME",
    "RELEASE_REPO",
    "RELEASE_TAG",
    "__version__",
]
