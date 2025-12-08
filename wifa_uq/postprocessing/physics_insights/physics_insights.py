# wifa_uq/postprocessing/physics_insights/physics_insights.py
"""
Physics Insights Module for WIFA-UQ.

Extracts interpretable physical insights from bias prediction models:
  1. Partial Dependence Analysis - How bias varies with each feature
  2. Feature Interactions - Which feature combinations drive error
  3. Regime Identification - Clustering of high-bias conditions
  4. Parameter-Condition Relationships - How optimal params depend on conditions

These analyses transform ML results into actionable physics understanding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class PartialDependenceResult:
    """Results from partial dependence analysis."""

    feature: str
    grid_values: np.ndarray
    pd_values: np.ndarray
    bias_direction: str  # "increases", "decreases", "non-monotonic"
    effect_magnitude: float  # Range of PD values
    physical_interpretation: str


@dataclass
class InteractionResult:
    """Results from interaction analysis."""

    feature_1: str
    feature_2: str
    interaction_strength: float
    description: str


@dataclass
class RegimeResult:
    """Results from regime identification."""

    regime_id: int
    n_cases: int
    mean_bias: float
    feature_centroids: dict[str, float]
    description: str


@dataclass
class ParameterRelationshipResult:
    """Results from parameter-condition analysis."""

    parameter: str
    most_influential_feature: str
    correlation: float
    relationship_type: str  # "positive", "negative", "weak"
    physical_interpretation: str


@dataclass
class PhysicsInsightsReport:
    """Complete physics insights report."""

    partial_dependence: list[PartialDependenceResult] = field(default_factory=list)
    interactions: list[InteractionResult] = field(default_factory=list)
    regimes: list[RegimeResult] = field(default_factory=list)
    parameter_relationships: list[ParameterRelationshipResult] = field(
        default_factory=list
    )
    summary: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "partial_dependence": [
                {
                    "feature": r.feature,
                    "bias_direction": r.bias_direction,
                    "effect_magnitude": r.effect_magnitude,
                    "interpretation": r.physical_interpretation,
                }
                for r in self.partial_dependence
            ],
            "interactions": [
                {
                    "features": [r.feature_1, r.feature_2],
                    "strength": r.interaction_strength,
                    "description": r.description,
                }
                for r in self.interactions
            ],
            "regimes": [
                {
                    "regime_id": r.regime_id,
                    "n_cases": r.n_cases,
                    "mean_bias": r.mean_bias,
                    "centroids": r.feature_centroids,
                    "description": r.description,
                }
                for r in self.regimes
            ],
            "parameter_relationships": [
                {
                    "parameter": r.parameter,
                    "most_influential_feature": r.most_influential_feature,
                    "correlation": r.correlation,
                    "interpretation": r.physical_interpretation,
                }
                for r in self.parameter_relationships
            ],
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = ["# Physics Insights Report\n"]

        if self.partial_dependence:
            lines.append("## 1. How Model Bias Depends on Atmospheric Conditions\n")
            for r in self.partial_dependence:
                lines.append(f"### {r.feature}")
                lines.append(
                    f"- **Direction**: Bias {r.bias_direction} with {r.feature}"
                )
                lines.append(f"- **Effect magnitude**: {r.effect_magnitude:.4f}")
                lines.append(f"- **Interpretation**: {r.physical_interpretation}\n")

        if self.interactions:
            lines.append("## 2. Feature Interactions\n")
            for r in self.interactions:
                lines.append(
                    f"- **{r.feature_1} × {r.feature_2}**: "
                    f"strength = {r.interaction_strength:.4f}"
                )
                lines.append(f"  - {r.description}\n")

        if self.regimes:
            lines.append("## 3. Error Regimes (Failure Modes)\n")
            for r in self.regimes:
                lines.append(f"### Regime {r.regime_id + 1}: {r.description}")
                lines.append(f"- Cases: {r.n_cases}")
                lines.append(f"- Mean bias: {r.mean_bias:.4f}")
                lines.append("- Characteristic conditions:")
                for feat, val in r.feature_centroids.items():
                    lines.append(f"  - {feat}: {val:.3f}")
                lines.append("")

        if self.parameter_relationships:
            lines.append("## 4. Optimal Parameter Dependencies\n")
            lines.append(
                "*How the 'best' wake model parameters vary with conditions:*\n"
            )
            for r in self.parameter_relationships:
                lines.append(f"### {r.parameter}")
                lines.append(
                    f"- Most influential feature: **{r.most_influential_feature}**"
                )
                lines.append(
                    f"- Correlation: {r.correlation:.3f} ({r.relationship_type})"
                )
                lines.append(f"- **Implication**: {r.physical_interpretation}\n")

        if self.summary:
            lines.append("## Summary\n")
            lines.append(self.summary)

        return "\n".join(lines)


# =============================================================================
# Physical Interpretation Helpers
# =============================================================================

# Domain knowledge for automatic interpretation
FEATURE_PHYSICS = {
    "ABL_height": {
        "high": "stable stratification / nocturnal conditions",
        "low": "convective / well-mixed conditions",
        "unit": "m",
    },
    "wind_veer": {
        "high": "strong directional shear / Ekman spiral",
        "low": "uniform wind direction with height",
        "unit": "deg/m",
    },
    "lapse_rate": {
        "high": "stable stratification (positive dθ/dz)",
        "low": "unstable/neutral conditions",
        "unit": "K/m",
    },
    "turbulence_intensity": {
        "high": "high ambient turbulence / faster wake recovery",
        "low": "low turbulence / slower wake recovery",
        "unit": "-",
    },
    "Blockage_Ratio": {
        "high": "significant upstream blockage",
        "low": "front-row or minimal blockage",
        "unit": "-",
    },
    "Blocking_Distance": {
        "high": "far from upstream turbines",
        "low": "close to upstream turbines",
        "unit": "-",
    },
    "Farm_Length": {
        "high": "deep array (many rows)",
        "low": "shallow array",
        "unit": "D",
    },
    "Farm_Width": {
        "high": "wide array",
        "low": "narrow array",
        "unit": "D",
    },
}

PARAMETER_PHYSICS = {
    "k_b": {
        "name": "Wake expansion coefficient",
        "increases_with": "faster wake recovery / higher turbulence",
        "decreases_with": "slower wake recovery / stable conditions",
    },
    "ss_alpha": {
        "name": "Self-similarity blockage parameter",
        "increases_with": "stronger blockage effects",
        "decreases_with": "weaker blockage effects",
    },
    "ceps": {
        "name": "Added turbulence coefficient",
        "increases_with": "more wake-added turbulence",
        "decreases_with": "less wake-added turbulence",
    },
}


def interpret_pd_direction(
    feature: str,
    direction: str,  # "increases", "decreases", "non-monotonic", or "flat"
) -> str:
    """Generate physical interpretation of partial dependence direction."""
    physics = FEATURE_PHYSICS.get(feature, {})
    high_meaning = physics.get("high", f"high {feature}")
    low_meaning = physics.get("low", f"low {feature}")

    if direction == "increases":
        return (
            f"Model bias increases with {feature}, suggesting the wake model "
            f"systematically underestimates wake effects in {high_meaning} "
            f"(and/or overestimates in {low_meaning})."
        )
    elif direction == "decreases":
        return (
            f"Model bias decreases with {feature}, suggesting the wake model "
            f"systematically overestimates wake effects in {high_meaning} "
            f"(and/or underestimates in {low_meaning})."
        )
    else:
        return (
            f"Model bias shows a non-monotonic relationship with {feature}, "
            f"suggesting different error mechanisms at {high_meaning} vs {low_meaning}."
        )


def interpret_parameter_relationship(
    parameter: str,
    feature: str,
    correlation: float,
) -> str:
    """Generate physical interpretation of parameter-feature relationship."""
    param_physics = PARAMETER_PHYSICS.get(parameter, {"name": parameter})
    feat_physics = FEATURE_PHYSICS.get(feature, {})

    param_name = param_physics.get("name", parameter)
    high_meaning = feat_physics.get("high", f"high {feature}")

    if abs(correlation) < 0.3:
        return f"{param_name} shows weak dependence on {feature}."

    if correlation > 0:
        return (
            f"{param_name} should increase in {high_meaning}. "
            f"This suggests the model's default parameterization underestimates "
            f"wake recovery rate in these conditions."
        )
    else:
        return (
            f"{param_name} should decrease in {high_meaning}. "
            f"This suggests the model's default parameterization overestimates "
            f"wake recovery rate in these conditions."
        )


def describe_regime(
    centroids: dict[str, float],
    feature_stats: dict[str, tuple[float, float]],  # {feature: (mean, std)}
    mean_bias: float,
) -> str:
    """Generate description of an error regime based on its characteristics."""
    descriptions = []

    for feature, value in centroids.items():
        if feature not in feature_stats:
            continue
        mean, std = feature_stats[feature]
        if std == 0:
            continue

        z_score = (value - mean) / std
        physics = FEATURE_PHYSICS.get(feature, {})

        if z_score > 1.0:
            descriptions.append(physics.get("high", f"high {feature}"))
        elif z_score < -1.0:
            descriptions.append(physics.get("low", f"low {feature}"))

    if not descriptions:
        if mean_bias > 0:
            return "Mixed conditions with positive bias (model overestimates)"
        else:
            return "Mixed conditions with negative bias (model underestimates)"

    bias_desc = "overestimation" if mean_bias > 0 else "underestimation"
    return f"{', '.join(descriptions[:2])} → {bias_desc}"


def describe_regime_relative(
    centroids: dict[str, float],
    all_cluster_info: list[dict],
    feature_stats: dict[str, tuple[float, float]],
    mean_bias: float,
    cluster_id: int,
) -> str:
    """
    Generate description of an error regime based on RELATIVE differences to other clusters.

    This is more informative when clusters are similar in absolute terms but differ
    from each other in specific ways.
    """
    # First try absolute description (z-scores > 1)
    descriptions = []
    for feature, value in centroids.items():
        if feature not in feature_stats:
            continue
        mean, std = feature_stats[feature]
        if std == 0:
            continue
        z_score = (value - mean) / std
        physics = FEATURE_PHYSICS.get(feature, {})
        if z_score > 1.0:
            descriptions.append(physics.get("high", f"high {feature}"))
        elif z_score < -1.0:
            descriptions.append(physics.get("low", f"low {feature}"))

    # If we found distinctive absolute features, use those
    if descriptions:
        bias_desc = "overestimation" if mean_bias > 0 else "underestimation"
        return f"{', '.join(descriptions[:2])} → {bias_desc}"

    # Otherwise, find what makes THIS cluster different from OTHERS
    other_clusters = [c for c in all_cluster_info if c["id"] != cluster_id]
    if not other_clusters:
        bias_desc = "overestimation" if mean_bias > 0 else "underestimation"
        return f"High-error cases → {bias_desc}"

    # Compute relative differences for each feature
    relative_diffs = []
    for feature, value in centroids.items():
        if feature not in feature_stats:
            continue
        mean, std = feature_stats[feature]
        if std == 0:
            continue

        # Average value in other clusters
        other_avg = np.mean([c["centroids"][feature] for c in other_clusters])
        diff = (value - other_avg) / std if std > 0 else 0
        relative_diffs.append((feature, diff, value, other_avg))

    # Sort by absolute difference
    relative_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Build description from top distinguishing features
    distinguishing = []
    for feature, diff, val, other_avg in relative_diffs[:2]:
        if abs(diff) < 0.3:  # Not meaningfully different
            continue
        physics = FEATURE_PHYSICS.get(feature, {})
        if diff > 0:
            desc = physics.get("high", f"higher {feature}")
            distinguishing.append(f"{desc} (vs other regimes)")
        else:
            desc = physics.get("low", f"lower {feature}")
            distinguishing.append(f"{desc} (vs other regimes)")

    bias_desc = "overestimation" if mean_bias > 0 else "underestimation"

    if distinguishing:
        return f"{'; '.join(distinguishing)} → {bias_desc}"

    # Last resort: describe by bias magnitude
    other_biases = [c["mean_bias"] for c in other_clusters]
    avg_other_bias = np.mean(other_biases)

    if abs(mean_bias) > abs(avg_other_bias) * 1.5:
        return f"Highest error magnitude ({mean_bias:.4f}) → {bias_desc}"
    elif abs(mean_bias) < abs(avg_other_bias) * 0.7:
        return f"Lower error magnitude ({mean_bias:.4f}) → {bias_desc}"
    else:
        return f"Similar conditions, bias={mean_bias:.4f} → {bias_desc}"


# =============================================================================
# Analysis Functions
# =============================================================================


def _manual_partial_dependence(
    model,
    X: pd.DataFrame,
    features: list[str],
    grid_resolution: int = 50,
) -> dict:
    """
    Manual partial dependence calculation as fallback.

    For each feature, creates a grid of values and computes mean prediction
    while averaging over all other features.
    """
    results = {
        "average": [],
        "grid_values": [],
    }

    X_array = X.values

    for feature in features:
        feat_idx = X.columns.get_loc(feature)
        feat_values = X[feature].values

        # Create grid
        grid = np.linspace(feat_values.min(), feat_values.max(), grid_resolution)

        # Compute partial dependence
        pd_values = []
        for grid_val in grid:
            # Create modified X with feature set to grid value
            X_modified = X_array.copy()
            X_modified[:, feat_idx] = grid_val

            # Predict and average
            predictions = model.predict(X_modified)
            pd_values.append(predictions.mean())

        results["average"].append(np.array(pd_values))
        results["grid_values"].append(grid)

    return results


def analyze_partial_dependence(
    model,
    X: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    grid_resolution: int = 50,
) -> list[PartialDependenceResult]:
    """
    Compute partial dependence and extract physical interpretations.

    Shows how predicted bias changes with each feature, holding others constant.
    """
    print("--- Analyzing Partial Dependence ---")
    results = []

    # Ensure X is a DataFrame with correct types
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.astype(float)

    # Verify model is fitted by trying a prediction
    try:
        _ = model.predict(X.iloc[:1])
    except Exception as e:
        print(f"    WARNING: Model does not appear to be fitted: {e}")
        print("    Attempting to re-fit model...")
        raise ValueError(
            f"Model must be fitted before partial dependence analysis: {e}"
        )

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, feature in enumerate(features):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Compute PD for this single feature (avoids sklearn multi-feature output issues)
        feature_idx = X.columns.get_loc(feature)

        try:
            pd_result = partial_dependence(
                model, X, features=[feature_idx], grid_resolution=grid_resolution
            )
            pd_values = pd_result["average"][0]
            grid_values = pd_result["grid_values"][0]
        except Exception as e:
            print(f"    WARNING: partial_dependence failed for {feature}: {e}")
            print("    Falling back to manual PD calculation...")
            manual_result = _manual_partial_dependence(
                model, X, [feature], grid_resolution
            )
            pd_values = manual_result["average"][0]
            grid_values = manual_result["grid_values"][0]

        # Flatten if needed (sklearn may return 2D arrays)
        pd_values = np.asarray(pd_values).flatten()
        grid_values = np.asarray(grid_values).flatten()

        # Determine direction
        n_points = len(pd_values)
        n_edge = min(5, n_points // 4) if n_points > 4 else 1

        start_val = pd_values[:n_edge].mean()
        end_val = pd_values[-n_edge:].mean()
        mid_val = pd_values[n_points // 2] if n_points > 2 else pd_values.mean()

        value_range = pd_values.max() - pd_values.min()
        threshold = 0.01 * value_range if value_range > 0 else 1e-10

        if end_val > start_val + threshold:
            direction = "increases"
        elif end_val < start_val - threshold:
            direction = "decreases"
        else:
            # Check for non-monotonicity
            if abs(mid_val - start_val) > abs(end_val - start_val):
                direction = "non-monotonic"
            else:
                direction = "flat"

        effect_magnitude = float(value_range) if value_range > 0 else 0.0
        interpretation = interpret_pd_direction(feature, direction)

        results.append(
            PartialDependenceResult(
                feature=feature,
                grid_values=grid_values,
                pd_values=pd_values,
                bias_direction=direction,
                effect_magnitude=float(effect_magnitude),
                physical_interpretation=interpretation,
            )
        )

        # Plot
        ax.plot(grid_values, pd_values, "b-", linewidth=2)
        ax.fill_between(grid_values, pd_values, alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)

        # Add direction annotation
        direction_symbol = {
            "increases": "↗",
            "decreases": "↘",
            "non-monotonic": "↝",
            "flat": "→",
        }
        ax.set_title(f"{feature} {direction_symbol.get(direction, '')}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial Dependence (Bias)")
        ax.grid(True, alpha=0.3)

        # Add interpretation as text box
        textstr = f"Effect: {effect_magnitude:.4f}\n{direction}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=props,
        )

    # Hide unused subplots
    for idx in range(n_features, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle("Partial Dependence: How Bias Varies with Each Feature", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"    Saved partial dependence plot to {output_dir / 'partial_dependence.png'}"
    )

    return results


def analyze_interactions(
    model,
    X: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    top_n: int = 5,
) -> list[InteractionResult]:
    """
    Analyze feature interactions using SHAP interaction values.

    Identifies which feature combinations jointly drive bias.
    """
    print("--- Analyzing Feature Interactions ---")

    if not HAS_SHAP:
        print("    SHAP not available, skipping interaction analysis")
        return []

    results = []

    try:
        # Get SHAP interaction values
        explainer = shap.TreeExplainer(model)
        shap_interaction = explainer.shap_interaction_values(
            X.values[:500]
        )  # Limit for speed

        # Average absolute interaction strength
        n_features = len(features)
        interaction_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    interaction_matrix[i, j] = np.abs(shap_interaction[:, i, j]).mean()

        # Find top interactions
        interactions_flat = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions_flat.append(
                    (
                        features[i],
                        features[j],
                        interaction_matrix[i, j] + interaction_matrix[j, i],
                    )
                )

        interactions_flat.sort(key=lambda x: x[2], reverse=True)

        for feat1, feat2, strength in interactions_flat[:top_n]:
            physics1 = FEATURE_PHYSICS.get(feat1, {}).get("high", f"high {feat1}")
            physics2 = FEATURE_PHYSICS.get(feat2, {}).get("high", f"high {feat2}")

            description = (
                f"Combined effect of {physics1} and {physics2} "
                f"creates bias beyond individual effects."
            )

            results.append(
                InteractionResult(
                    feature_1=feat1,
                    feature_2=feat2,
                    interaction_strength=float(strength),
                    description=description,
                )
            )

        # Plot interaction heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(interaction_matrix, cmap="YlOrRd")

        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_yticklabels(features)

        # Add values
        for i in range(n_features):
            for j in range(n_features):
                val = interaction_matrix[i, j]
                color = "white" if val > interaction_matrix.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        plt.colorbar(im, ax=ax, label="Interaction Strength")
        ax.set_title(
            "Feature Interaction Strengths\n(Higher = stronger combined effect on bias)"
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "feature_interactions.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(
            f"    Saved interaction plot to {output_dir / 'feature_interactions.png'}"
        )

    except Exception as e:
        print(f"    Interaction analysis failed: {e}")

    return results


def analyze_regimes(
    X: pd.DataFrame,
    y_bias: np.ndarray,
    features: list[str],
    output_dir: Path,
    n_clusters: int = 3,
    bias_percentile: float = 75,
) -> list[RegimeResult]:
    """
    Identify distinct error regimes through clustering.

    Clusters high-bias cases to find systematic failure modes.
    """
    print("--- Analyzing Error Regimes ---")
    results = []

    # Focus on high-bias cases
    bias_threshold = np.percentile(np.abs(y_bias), bias_percentile)
    high_bias_mask = np.abs(y_bias) >= bias_threshold

    X_high = X[high_bias_mask].copy()
    y_high = y_bias[high_bias_mask]

    # Hard minimum: need at least 10 high-bias cases
    if len(X_high) < 10:
        print(
            f"    WARNING: Only {len(X_high)} high-bias cases. Need at least 10 for meaningful regime analysis."
        )
        print(
            "    Skipping regime clustering. Consider lowering bias_percentile or getting more data."
        )

        # Still report the high-bias cases as a single "regime" for visibility
        if len(X_high) > 0:
            centroids = {f: float(X_high[f].mean()) for f in features}
            mean_bias = float(y_high.mean())
            feature_stats = {f: (X[f].mean(), X[f].std()) for f in features}

            results.append(
                RegimeResult(
                    regime_id=0,
                    n_cases=len(X_high),
                    mean_bias=mean_bias,
                    feature_centroids=centroids,
                    description=f"All {len(X_high)} high-bias cases (too few for clustering). Mean bias: {mean_bias:.4f}",
                )
            )

            # Create simple summary plot
            fig, ax = plt.subplots(figsize=(10, 5))

            z_scores = []
            for f in features:
                mean, std = feature_stats[f]
                z = (centroids[f] - mean) / std if std > 0 else 0
                z_scores.append(z)

            colors = ["red" if abs(z) > 1 else "steelblue" for z in z_scores]
            ax.barh(features, z_scores, color=colors)
            ax.axvline(0, color="k", linestyle="-", linewidth=0.5)
            ax.axvline(-1, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(1, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Z-Score (deviation from dataset mean)")
            ax.set_title(
                f"High-Bias Cases Characteristics (n={len(X_high)})\nRed = distinctive (|Z| > 1)"
            )

            plt.tight_layout()
            plt.savefig(output_dir / "error_regimes.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved high-bias summary to {output_dir / 'error_regimes.png'}")

        return results

    # Adjust n_clusters based on available data
    # Rule: at least 5 cases per cluster
    max_clusters = len(X_high) // 5
    if max_clusters < n_clusters:
        print(
            f"    Reducing clusters from {n_clusters} to {max_clusters} (need 5+ cases per cluster)"
        )
        n_clusters = max(2, max_clusters)

    # Standardize for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_high[features])

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Compute feature statistics for interpretation
    feature_stats = {f: (X[f].mean(), X[f].std()) for f in features}

    # First pass: collect all cluster info
    cluster_info = []
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_X = X_high[mask]
        cluster_y = y_high[mask]
        centroids = {f: float(cluster_X[f].mean()) for f in features}
        mean_bias = float(cluster_y.mean())
        cluster_info.append(
            {
                "id": cluster_id,
                "n_cases": int(mask.sum()),
                "mean_bias": mean_bias,
                "centroids": centroids,
            }
        )

    # Second pass: describe regimes using RELATIVE differences between clusters
    for info in cluster_info:
        description = describe_regime_relative(
            info["centroids"],
            cluster_info,
            feature_stats,
            info["mean_bias"],
            info["id"],
        )

        results.append(
            RegimeResult(
                regime_id=info["id"],
                n_cases=info["n_cases"],
                mean_bias=info["mean_bias"],
                feature_centroids=info["centroids"],
                description=description,
            )
        )

    # Plot regime analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PCA projection of clusters
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6, s=30
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].set_title(f"High-Bias Cases Clustered into {n_clusters} Regimes")
    plt.colorbar(scatter, ax=axes[0], label="Regime")

    # Right: Regime characteristics bar chart
    regime_data = []
    for r in results:
        for feat, val in r.feature_centroids.items():
            mean, std = feature_stats[feat]
            z_score = (val - mean) / std if std > 0 else 0
            regime_data.append(
                {
                    "Regime": f"Regime {r.regime_id + 1}\n({r.n_cases} cases)",
                    "Feature": feat,
                    "Z-Score": z_score,
                }
            )

    regime_df = pd.DataFrame(regime_data)
    regime_pivot = regime_df.pivot(index="Feature", columns="Regime", values="Z-Score")

    regime_pivot.plot(kind="barh", ax=axes[1], width=0.8)
    axes[1].axvline(0, color="k", linestyle="-", linewidth=0.5)
    axes[1].axvline(-1, color="r", linestyle="--", alpha=0.5, linewidth=0.5)
    axes[1].axvline(1, color="r", linestyle="--", alpha=0.5, linewidth=0.5)
    axes[1].set_xlabel("Z-Score (deviation from mean)")
    axes[1].set_title("Regime Characteristics\n(|Z| > 1 = distinctive)")
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "error_regimes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved regime analysis to {output_dir / 'error_regimes.png'}")

    return results


def analyze_parameter_relationships(
    database: xr.Dataset,
    calibrator,
    features: list[str],
    output_dir: Path,
) -> list[ParameterRelationshipResult]:
    """
    Analyze how optimal parameters depend on atmospheric conditions.

    Only applicable for local calibration where optimal params vary per case.
    """
    print("--- Analyzing Parameter-Condition Relationships ---")
    results = []

    # Check if we have local calibration results
    if not hasattr(calibrator, "optimal_params_") or calibrator.optimal_params_ is None:
        print("    No local calibration results available")
        return results

    swept_params = calibrator.swept_params
    optimal_params = calibrator.optimal_params_

    # Get feature values (from sample=0, features don't depend on sample)
    X_df = database.isel(sample=0).to_dataframe().reset_index()
    X = X_df[features]

    n_params = len(swept_params)
    n_features = len(features)

    fig, axes = plt.subplots(
        n_params, n_features, figsize=(4 * n_features, 4 * n_params)
    )
    if n_params == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)

    for p_idx, param in enumerate(swept_params):
        param_values = optimal_params[param]

        best_feature = None
        best_corr = 0

        for f_idx, feature in enumerate(features):
            ax = axes[p_idx, f_idx]
            feature_values = X[feature].values

            # Compute correlation
            corr = np.corrcoef(feature_values, param_values)[0, 1]
            if np.isnan(corr):
                corr = 0

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_feature = feature

            # Scatter plot with regression line
            ax.scatter(feature_values, param_values, alpha=0.4, s=10)

            # Add regression line
            if abs(corr) > 0.1:
                z = np.polyfit(feature_values, param_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(feature_values.min(), feature_values.max(), 100)
                ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r={corr:.2f}")
                ax.legend(fontsize=8)

            ax.set_xlabel(feature)
            if f_idx == 0:
                ax.set_ylabel(f"Optimal {param}")
            ax.set_title(f"r = {corr:.3f}")
            ax.grid(True, alpha=0.3)

        # Store result for best correlated feature
        if best_feature:
            if abs(best_corr) > 0.5:
                rel_type = "strong positive" if best_corr > 0 else "strong negative"
            elif abs(best_corr) > 0.3:
                rel_type = "moderate positive" if best_corr > 0 else "moderate negative"
            else:
                rel_type = "weak"

            interpretation = interpret_parameter_relationship(
                param, best_feature, best_corr
            )

            results.append(
                ParameterRelationshipResult(
                    parameter=param,
                    most_influential_feature=best_feature,
                    correlation=float(best_corr),
                    relationship_type=rel_type,
                    physical_interpretation=interpretation,
                )
            )

    plt.suptitle(
        "Optimal Parameter vs. Atmospheric Conditions\n"
        "(Shows how model params should vary with conditions)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "parameter_relationships.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"    Saved parameter relationships to {output_dir / 'parameter_relationships.png'}"
    )

    return results


def generate_summary(report: PhysicsInsightsReport) -> str:
    """Generate executive summary of physics insights."""
    lines = []

    # Summarize PD findings
    if report.partial_dependence:
        increasing = [
            r for r in report.partial_dependence if r.bias_direction == "increases"
        ]
        decreasing = [
            r for r in report.partial_dependence if r.bias_direction == "decreases"
        ]

        if increasing:
            feats = ", ".join(r.feature for r in increasing[:2])
            lines.append(f"• Model bias increases with {feats}")
        if decreasing:
            feats = ", ".join(r.feature for r in decreasing[:2])
            lines.append(f"• Model bias decreases with {feats}")

    # Summarize interactions
    if report.interactions:
        top = report.interactions[0]
        lines.append(f"• Strongest interaction: {top.feature_1} × {top.feature_2}")

    # Summarize regimes
    if report.regimes:
        n_regimes = len(report.regimes)
        lines.append(f"• Identified {n_regimes} distinct error regimes")
        for r in report.regimes:
            lines.append(f"  - Regime {r.regime_id + 1}: {r.description}")

    # Summarize parameter insights
    if report.parameter_relationships:
        for r in report.parameter_relationships:
            if "strong" in r.relationship_type:
                lines.append(
                    f"• {r.parameter} should be condition-dependent "
                    f"(varies with {r.most_influential_feature})"
                )

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================


def run_physics_insights(
    database: xr.Dataset,
    fitted_model,
    calibrator,
    features_list: list[str],
    y_bias: np.ndarray,
    output_dir: Path,
    config: dict | None = None,
) -> PhysicsInsightsReport:
    """
    Run complete physics insights analysis.

    Args:
        database: xarray Dataset with model error database
        fitted_model: Trained ML model (or pipeline with 'model' step)
        calibrator: Fitted calibrator (for parameter relationships)
        features_list: List of feature names used
        y_bias: Bias values (predictions or actuals)
        output_dir: Directory for output plots
        config: Optional configuration dict

    Returns:
        PhysicsInsightsReport with all analysis results
    """
    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHYSICS INSIGHTS ANALYSIS")
    print("=" * 60)

    # Prepare feature matrix
    X_df = database.isel(sample=0).to_dataframe().reset_index()
    X = X_df[features_list]

    # Extract model from pipeline if needed
    if hasattr(fitted_model, "named_steps") and "model" in fitted_model.named_steps:
        model = fitted_model.named_steps["model"]
    else:
        model = fitted_model

    report = PhysicsInsightsReport()

    # 1. Partial Dependence
    if config.get("partial_dependence", {}).get("enabled", True):
        pd_features = config.get("partial_dependence", {}).get(
            "features", features_list
        )
        report.partial_dependence = analyze_partial_dependence(
            model=fitted_model,  # Use full pipeline for PD
            X=X,
            features=pd_features,
            output_dir=output_dir,
        )

    # 2. Interactions (requires tree model)
    if config.get("interactions", {}).get("enabled", True):
        if hasattr(model, "feature_importances_"):  # Tree-based model
            report.interactions = analyze_interactions(
                model=model,
                X=X,
                features=features_list,
                output_dir=output_dir,
                top_n=config.get("interactions", {}).get("top_n", 5),
            )
        else:
            print("    Skipping interactions (requires tree-based model)")

    # 3. Regime Analysis
    if config.get("regime_analysis", {}).get("enabled", True):
        report.regimes = analyze_regimes(
            X=X,
            y_bias=y_bias,
            features=features_list,
            output_dir=output_dir,
            n_clusters=config.get("regime_analysis", {}).get("n_clusters", 3),
            bias_percentile=config.get("regime_analysis", {}).get(
                "bias_percentile", 75
            ),
        )

    # 4. Parameter Relationships (local calibration only)
    if config.get("parameter_relationships", {}).get("enabled", True):
        if hasattr(calibrator, "optimal_params_"):
            report.parameter_relationships = analyze_parameter_relationships(
                database=database,
                calibrator=calibrator,
                features=features_list,
                output_dir=output_dir,
            )

    # Generate summary
    report.summary = generate_summary(report)

    # Save report
    report_md = report.to_markdown()
    with open(output_dir / "physics_insights_report.md", "w") as f:
        f.write(report_md)
    print(f"\nSaved report to {output_dir / 'physics_insights_report.md'}")

    # Save as JSON for programmatic access
    import json

    with open(output_dir / "physics_insights.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved JSON to {output_dir / 'physics_insights.json'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(report.summary)

    return report
