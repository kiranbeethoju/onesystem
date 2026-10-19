"""Colab helper — prefer copy-paste from medlm/COLAB.md

This module documents the GPU train entrypoints; run cells from COLAB.md in Colab.
"""

COLAB_CLONE = """
!git clone --depth 1 -b onesystem https://github.com/kiranbeethoju/onesystem.git
%cd /content/onesystem
!pip -q install -U torch datasets "huggingface_hub>=0.34,<1" tokenizers safetensors tqdm
import os, torch
os.environ["PYTHONPATH"] = "/content/onesystem"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
"""

COLAB_DOWNLOAD = """
!python -m medlm.download_public_corpora --out data/medlm/public
!python -m medlm.prepare_corpus --bootstrap 2000 --humigy
!python -m medlm.build_tokenizer --vocab-size 16000
!python -m medlm.encode_data
"""

COLAB_TRAIN_L4 = """
!python -m medlm.train --device cuda --max-steps 10000 \\
  --batch-size 8 --grad-accum 4 --block-size 512 \\
  --warmup-steps 200 --eval-every 200 --save-every 500 \\
  --out models/medlm-120m
"""
