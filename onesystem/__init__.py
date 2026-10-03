"""OneSystem: a local specialist model for operational decisions."""

__version__ = "0.1.0"

MODEL_NAME = "OneSystem"
BASE_CHECKPOINT = "fastino/GLiNER2.5-Decide"
DATASET_ID = "fastino/fast-decisions"

__all__ = ["BASE_CHECKPOINT", "DATASET_ID", "MODEL_NAME", "__version__"]
