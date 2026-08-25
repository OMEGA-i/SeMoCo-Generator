"""SeMoCo-Generator — an autoregressive text-to-motion LM over SeMoCo tokens.

Time-axis autoregression over low-frequency (12.5Hz) motion packets, with
codebook-axis multi-token prediction (8 RVQ heads). The motion tokenizer
(SeMoCo) stays frozen; this package only consumes its codes.
"""

__version__ = "1.0.0"
