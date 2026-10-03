"""OneSystem: a standalone System 1 decision model.

OneSystem reads a document together with the labels that are legal for a
decision and scores every label in one forward pass. It is its own model: a
bidirectional encoder plus a label-scoring head, saved as plain safetensors.
It does not load or depend on GLiNER checkpoints.
"""

__version__ = "0.2.0"

MODEL_NAME = "OneSystem"
# Encoder used to initialise training. After training, OneSystem ships the full
# encoder weights under its own name and never downloads this checkpoint again.
DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"
DATASET_ID = "fastino/fast-decisions"
RELEASE_REPO = "kiranbeethoju/onesystem"
RELEASE_TAG = "v0.2.0"

__all__ = [
    "DATASET_ID",
    "DEFAULT_ENCODER",
    "MODEL_NAME",
    "RELEASE_REPO",
    "RELEASE_TAG",
    "__version__",
]
