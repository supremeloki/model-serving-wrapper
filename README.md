# model-serving-wrapper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A model serving wrapper: input validation, prediction caching, warmup, and A/B rollout — production serving concerns around any `predict()` function.

## 🚀 Overview

The gap between a trained model and a *served* model is operational: validating inputs, warming the inference path, caching repeat predictions, and safely testing new versions. `model-serving-wrapper` wraps any model in a `ModelServer` that enforces required fields (naming exactly what's missing), caches per-entity predictions with bounded size, and supports **A/B rollouts** where a failing treatment transparently falls back to control.

## ✨ Features

- **Input validation:** required fields enforced; violations list the exact missing names
- **Load discipline:** predict-before-load rejected; zero-weight models caught at load time
- **Prediction cache:** entity-keyed, FIFO-bounded, hit-flagged on every `Prediction`
- **Warmup contract:** models pre-execute their inference path to pay JIT/load costs up front
- **ABRollout:** percentage-based traffic split with automatic control fallback on treatment failure
- **Two reference models:** rule-based dispatch and linear scorer included for testing
- **Zero dependencies**

## 🚧 Structure

```
model-serving-wrapper/
├── src/model_serving/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/model-serving-wrapper.git
cd model-serving-wrapper
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from model_serving import LinearScorerModel, ModelServer

churn_model = LinearScorerModel(
    weights={"tenure": 0.8, "spend": 1.2}, bias=0.5,
)
server = ModelServer(churn_model, required_fields=("tenure",))
server.load_model()

prediction = server.predict({"tenure": 10, "spend": 100}, entity_key="u:42")
print(prediction.value, prediction.from_cache)
```

### A/B rollout

```python
from model_serving import ABRollout

rollout = ABRollout(control_server, candidate_server, treatment_share_percent=10)
verdict = rollout.route(payload)
```

## 🔧 Error Handling

```text
ServingError
├── ModelNotLoadedError    # predict() before load()
└── InvalidInputError      # .missing lists absent required fields
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen predictions
- Zero comments — names carry the meaning
- Rollout fallback tested against a deliberately broken treatment

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
