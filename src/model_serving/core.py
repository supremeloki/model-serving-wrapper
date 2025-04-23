from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


class ServingError(Exception):
    pass


class ModelNotLoadedError(ServingError):
    pass


class InvalidInputError(ServingError):
    def __init__(self, missing: Sequence[str]) -> None:
        super().__init__(f"missing required fields: {sorted(missing)}")
        self.missing = tuple(missing)


@dataclass(frozen=True)
class Prediction:
    value: Any
    model_version: str
    latency_ms: float
    from_cache: bool = False

    @property
    def is_warm(self) -> bool:
        return not self.from_cache and self.latency_ms < 50.0


class LoadedModel(Protocol := object):
    pass


class BaseModelWrapper:
    version: str = "0.0.0"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, features: dict[str, Any]) -> Any: ...

    @abstractmethod
    def warmup(self) -> None: ...


class RuleBasedModel(BaseModelWrapper):
    """Placeholder docstring removed"""

    version = "rules-1.0"

    def __init__(self, rules: dict[str, Callable[[dict[str, Any]], Any]]) -> None:
        self._rules = rules
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def predict(self, features: dict[str, Any]) -> Any:
        if not self._loaded:
            raise ModelNotLoadedError("call load() before predict()")
        for name, rule in self._rules.items():
            if name in features:
                return rule(features[name])
        raise ServingError(f"no rule matched any of {sorted(features)}")

    def warmup(self) -> None:
        self.load()


class LinearScorerModel(BaseModelWrapper):
    version = "linear-1.2"

    def __init__(self, weights: dict[str, float], bias: float = 0.0) -> None:
        self._weights = weights
        self._bias = bias
        self._loaded = False

    def load(self) -> None:
        total = sum(abs(w) for w in self._weights.values())
        if total == 0.0:
            raise ServingError("model has zero weights")
        self._loaded = True

    def predict(self, features: dict[str, Any]) -> float:
        if not self._loaded:
            raise ModelNotLoadedError()
        score = self._bias + sum(
            weight * float(features.get(name, 0.0))
            for name, weight in self._weights.items()
        )
        return round(score, 6)

    def warmup(self) -> None:
        dummy = {name: 0.0 for name in self._weights}
        self.predict(dummy)


class ModelServer:
    def __init__(self, wrapper: BaseModelWrapper,
                 required_fields: Sequence[str] = (),
                 cache_predictions: bool = True,
                 max_cache_entries: int = 512) -> None:
        self._wrapper = wrapper
        self._required = tuple(required_fields)
        self._cache_enabled = cache_predictions
        self._cache: dict[str, Any] = {}
        self._max_cache = max_cache_entries
        self.request_count = 0

    @property
    def model_version(self) -> str:
        return self._wrapper.version

    def load_model(self) -> None:
        self._wrapper.load()

    def predict(self, payload: dict[str, Any],
                entity_key: str | None = None) -> Prediction:
        missing = [name for name in self._required if name not in payload]
        if missing:
            raise InvalidInputError(missing)

        started = time.perf_counter()
        cache_key = f"{entity_key}::{sorted(payload.items())}" if entity_key else None
        if self._cache_enabled and cache_key and cache_key in self._cache:
            elapsed = (time.perf_counter() - started) * 1000
            return Prediction(value=self._cache[cache_key],
                              model_version=self.model_version,
                              latency_ms=round(elapsed, 3), from_cache=True)

        raw_value = self._wrapper.predict(payload)
        elapsed = (time.perf_counter() - started) * 1000
        if self._cache_enabled and cache_key:
            if len(self._cache) >= self._max_cache:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = raw_value
        self.request_count += 1
        return Prediction(
            value=raw_value,
            model_version=self.model_version,
            latency_ms=round(elapsed, 3),
        )

    def warm_up(self) -> None:
        self._wrapper.warmup()

    def cache_size(self) -> int:
        return len(self._cache)


class ABRollout:
    def __init__(self, control: ModelServer, treatment: ModelServer,
                 treatment_share_percent: int = 10, seed: int = 7) -> None:
        import random as _random

        if not 0 <= treatment_share_percent <= 100:
            raise ServingError("treatment_share_percent must be within 0..100")
        self.control = control
        self.treatment = treatment
        self._share = treatment_share_percent
        self._rng = _random.Random(seed)

    def route(self, payload: dict[str, Any]) -> Prediction:
        bucket = self._rng.randint(1, 100)
        server = self.treatment if bucket <= self._share else self.control
        try:
            return server.predict(payload)
        except ServingError:
            fallback = self.control
            return fallback.predict(payload)
