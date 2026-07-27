"""
Feature Engineering Module for ManufacturingIQ

This module provides a single source of truth for all engineered features
used in both training (notebook) and inference (API).
"""

from .engineer import build_engineered_features

__all__ = ["build_engineered_features"]