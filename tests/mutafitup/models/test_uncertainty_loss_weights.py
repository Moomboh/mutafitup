import math

import torch

from mutafitup.models.uncertainty_loss_weights import UncertaintyLossWeights


def test_regression_formula_at_init():
    """At init (log_var=0, sigma^2=1), regression weight is 0.5."""
    ulw = UncertaintyLossWeights(
        task_names=["reg"],
        problem_types={"reg": "regression"},
    )
    loss = torch.tensor(2.0)
    weighted = ulw(loss, "reg")
    # 0.5 * exp(0) * 2.0 + 0.5 * 0 = 1.0
    assert torch.isclose(weighted, torch.tensor(1.0))


def test_classification_formula_at_init():
    """At init (log_var=0, sigma^2=1), classification weight is 1.0."""
    ulw = UncertaintyLossWeights(
        task_names=["cls"],
        problem_types={"cls": "classification"},
    )
    loss = torch.tensor(3.0)
    weighted = ulw(loss, "cls")
    # exp(0) * 3.0 + 0 = 3.0
    assert torch.isclose(weighted, torch.tensor(3.0))


def test_regression_formula_with_nonzero_log_var():
    ulw = UncertaintyLossWeights(
        task_names=["reg"],
        problem_types={"reg": "regression"},
    )
    # Set log_var to ln(2) => sigma^2 = 2
    with torch.no_grad():
        ulw.log_vars["reg"].fill_(math.log(2.0))

    loss = torch.tensor(4.0)
    weighted = ulw(loss, "reg")
    # 0.5 * exp(-ln(2)) * 4.0 + 0.5 * ln(2) = 0.5 * 0.5 * 4.0 + 0.5 * ln(2)
    expected = 1.0 + 0.5 * math.log(2.0)
    assert torch.isclose(weighted, torch.tensor(expected))


def test_classification_formula_with_nonzero_log_var():
    ulw = UncertaintyLossWeights(
        task_names=["cls"],
        problem_types={"cls": "classification"},
    )
    with torch.no_grad():
        ulw.log_vars["cls"].fill_(math.log(2.0))

    loss = torch.tensor(3.0)
    weighted = ulw(loss, "cls")
    # exp(-ln(2)) * 3.0 + ln(2) = 0.5 * 3.0 + ln(2)
    expected = 1.5 + math.log(2.0)
    assert torch.isclose(weighted, torch.tensor(expected))


def test_multiple_tasks():
    ulw = UncertaintyLossWeights(
        task_names=["cls", "reg"],
        problem_types={"cls": "classification", "reg": "regression"},
    )
    cls_loss = ulw(torch.tensor(1.0), "cls")
    reg_loss = ulw(torch.tensor(1.0), "reg")

    # At init: cls = exp(0)*1 + 0 = 1.0, reg = 0.5*exp(0)*1 + 0 = 0.5
    assert torch.isclose(cls_loss, torch.tensor(1.0))
    assert torch.isclose(reg_loss, torch.tensor(0.5))


def test_parameters_are_learnable():
    ulw = UncertaintyLossWeights(
        task_names=["a", "b"],
        problem_types={"a": "classification", "b": "regression"},
    )
    params = list(ulw.parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


def test_gradient_flows():
    ulw = UncertaintyLossWeights(
        task_names=["reg"],
        problem_types={"reg": "regression"},
    )
    loss = torch.tensor(2.0, requires_grad=True)
    weighted = ulw(loss, "reg")
    weighted.backward()

    assert ulw.log_vars["reg"].grad is not None
    assert loss.grad is not None
