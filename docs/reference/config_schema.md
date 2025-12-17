# Configuration Schema

WIFA-UQ configuration is validated using Pydantic. The schema is defined in [`wifa_uq/workflow_schema.py`](https://github.com/EUFLOW/WIFA-UQ/blob/main/wifa_uq/workflow_schema.py).

## Validating a Config File

```python
import yaml
from wifa_uq.workflow_schema import WifaUQConfig

with open("my_config.yaml") as f:
    raw = yaml.safe_load(f)

config = WifaUQConfig(**raw)
```

Invalid configurations fail immediately with clear error messages:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for WifaUQConfig
error_prediction -> model
  Input should be 'XGB', 'PCE', 'SIRPolynomial' or 'Linear' [type=literal_error]
```

## Viewing the Schema

Generate a JSON Schema for editor integration or external tooling:

```python
from wifa_uq.workflow_schema import WifaUQConfig

print(WifaUQConfig.model_json_schema())
```
