"""App package init.

We intentionally disable TensorFlow / JAX backends inside ``transformers`` (used
transitively by ``sentence-transformers``) so the import path stays light and
doesn't pull in TF/Keras when those frameworks happen to be installed in the
same environment. We only need the PyTorch path.
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
