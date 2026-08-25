"""SOMA/TMR protocol defaults."""
FPS = 30.0
# Retrieval gallery size: "batch32", "batch256" or "full_gallery".
RETRIEVAL_PROTOCOL = "batch32"
TMR_MODEL = "tmr-soma-rp"
# Model-native FPS defaults when meta.json is missing.
MODEL_FPS = {
    "semoco": 50.0,
}
__all__ = ["FPS", "MODEL_FPS", "RETRIEVAL_PROTOCOL", "TMR_MODEL"]
