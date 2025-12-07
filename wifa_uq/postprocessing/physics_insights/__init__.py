# wifa_uq/postprocessing/physics_insights/__init__.py
"""
Physics Insights Module.

Extracts interpretable physical insights from bias prediction models.
"""

from .physics_insights import (
    run_physics_insights,
    PhysicsInsightsReport,
    PartialDependenceResult,
    InteractionResult,
    RegimeResult,
    ParameterRelationshipResult,
)

__all__ = [
    "run_physics_insights",
    "PhysicsInsightsReport",
    "PartialDependenceResult",
    "InteractionResult",
    "RegimeResult",
    "ParameterRelationshipResult",
]
