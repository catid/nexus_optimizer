"""Nexus optimizer: Algorithm 3 with DDP-safe normalized inner steps."""

from .optimizer import NexusOptimizer, NormalizedSGD

__version__ = "0.1.0"

__all__ = ["NexusOptimizer", "NormalizedSGD"]
