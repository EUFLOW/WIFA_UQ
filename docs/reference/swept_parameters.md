# Available Swept Parameters

Parameters that can be calibrated via the `param_config` section.

## Wake Deficit Models

### Bastankhah Gaussian (`Bastankhah2014`)

| Parameter | Path | Typical Range | Default |
|-----------|------|---------------|---------|
| Wake expansion (ambient) | `...wake_expansion_coefficient.k_a` | [0.01, 0.1] | 0.04 |
| Wake expansion (TI) | `...wake_expansion_coefficient.k_b` | [0.01, 0.07] | 0.04 |
| Epsilon coefficient | `...ceps` | [0.15, 0.3] | 0.2154 |

### Jensen (`Jensen1983`)

| Parameter | Path | Typical Range | Default |
|-----------|------|---------------|---------|
| Wake decay | `...wake_decay_coefficient` | [0.04, 0.1] | 0.075 |

## Blockage Models

### Self-Similarity (`SelfSimilarityDeficit2020`)

| Parameter | Path | Typical Range | Default |
|-----------|------|---------------|---------|
| Alpha | `...ss_alpha` | [0.75, 1.0] | 0.875 |

## Adding Custom Parameters

Any parameter accessible via windIO path notation can be swept:
```yaml
param_config:
  attributes.analysis.your_model.your_param:
    range: [min, max]
    default: nominal_value
    short_name: display_name
```
