import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from model_serving import (
    ABRollout,
    InvalidInputError,
    ModelNotLoadedError,
    ModelServer,
    RuleBasedModel,
    ServingError,
)


@pytest.fixture
def linear_model():
    model = __import__("model_serving.core", fromlist=["LinearScorerModel"]).LinearScorerModel(
        weights={"tenure": 0.8, "spend": 1.2}, bias=0.5
    )
    return model


def make_server(linear_model, required=("tenure",)):
    return ModelServer(linear_model, required_fields=required)


def test_predict_before_load_rejected(linear_model):
    server = make_server(linear_model)
    with pytest.raises(ModelNotLoadedError):
        server.predict({"tenure": 5})


def test_missing_required_field_lists_names(linear_model):
    server = make_server(linear_model)
    with pytest.raises(InvalidInputError) as excinfo:
        server.load_model() or server.predict({"spend": 3})
    assert "tenure" in excinfo.value.missing


def test_linear_scoring_math(linear_model):
    server = make_server(linear_model)
    server.load_model()
    prediction = server.predict({"tenure": 10, "spend": 100})
    expected = round(0.5 + 0.8 * 10 + 1.2 * 100, 6)
    assert prediction.value == pytest.approx(expected)
    assert not prediction.from_cache


def test_prediction_cached_on_entity_key(linear_model):
    server = make_server(linear_model)
    server.load_model()
    first = server.predict({"tenure": 1}, entity_key="u:9")
    second = server.predict({"tenure": 1}, entity_key="u:9")
    assert first.from_cache is False
    assert second.from_cache is True
    assert first.value == second.value


def test_cache_respects_max_entries(linear_model):
    server = ModelServer(linear_model, max_cache_entries=2)
    server.load_model()
    for i in range(4):
        server.predict({"tenure": i}, entity_key=f"e{i}")
    assert server.cache_size() == 2


def test_rule_based_model_dispatch():
    rules = {"country": lambda c: f"region-{c}", "tier": lambda t: t.upper()}
    model = RuleBasedModel(rules)
    model.load()
    assert model.predict({"country": "ir"}) == "region-ir"
    assert model.predict({"tier": "gold"}) == "GOLD"


def test_zero_weight_model_rejected_at_load():
    core = __import__("model_serving.core", fromlist=["LinearScorerModel"])
    model = core.LinearScorerModel(weights={"a": 0.0})
    with pytest.raises(ServingError):
        model.load()


def test_warmup_runs_inference_path(linear_model):
    server = make_server(linear_model)
    server.load_model()
    server.warm_up()
    assert server.model_version == "linear-1.2"
    prediction = server.predict({"tenure": 5})
    assert prediction.value is not None


def test_ab_rollout_always_control_at_zero_share(linear_model):
    control = make_server(linear_model)
    control.load_model()
    treatment = ModelServer(RuleBasedModel({}), cache_predictions=False)
    rollout = ABRollout(control, treatment, treatment_share_percent=0)
    for _ in range(20):
        prediction = rollout.route({"tenure": 3})
        assert prediction.model_version == control.model_version


def test_rollout_falls_back_when_treatment_fails():
    good_model = __import__("model_serving.core", fromlist=["LinearScorerModel"]).LinearScorerModel(
        weights={"x": 1.0}
    )
    control = ModelServer(good_model)
    control.load_model()

    class ExplodingWrapper:
        version = "broken"

        def load(self): ...
        def warmup(self): ...

        def predict(self, features):
            raise ServingError("treatment down")

    broken_server = ModelServer(ExplodingWrapper())
    broken_server.load_model()
    rollout = ABRollout(broken_server, control, treatment_share_percent=50)
    prediction = rollout.route({"x": 2})
    assert prediction.value is not None
