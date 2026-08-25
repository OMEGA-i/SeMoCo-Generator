from .base import TextEncoder
from .flan_t5_encoder import FlanT5Encoder
from .siglip_encoder import SigLIPEncoder
from .qwen3_encoder import Qwen3Encoder
from .registry import register, get_encoder_cls, list_encoders

# Register built-in text encoders.  New encoders are added here and in their
# own modules; callers use ``get_encoder_cls(key)`` instead of importing
# concrete classes directly.
register("flan", FlanT5Encoder)
register("siglip", SigLIPEncoder)
register("qwen3", Qwen3Encoder)

__all__ = [
    "TextEncoder",
    "FlanT5Encoder",
    "register",
    "get_encoder_cls",
    "list_encoders",
]
