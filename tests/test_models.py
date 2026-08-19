import torch

from adl_repro.models import FactorizedClusterLayer, ModelConfig, build_model


def test_all_models_forward_and_backward():
    categorical = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 1]])
    dense = torch.randn(4, 2)
    scenario = torch.tensor([0, 1, 2, 0])
    config = ModelConfig([10, 10], dense_dim=2, scenario_num=3, embedding_dim=4, hidden_dims=(16, 8), cluster_num=3)
    for name in ["sharedbottom", "adasparse", "adl"]:
        model = build_model(name, config)
        output = model(categorical, dense, scenario)
        assert output.logits.shape == (4,)
        assert output.auxiliary_loss.ndim == 0
        (output.logits.sum() + output.auxiliary_loss).backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_adasparse_reports_and_regularizes_sparsity():
    model = build_model(
        "adasparse",
        ModelConfig(
            [10, 10],
            dense_dim=1,
            scenario_num=3,
            embedding_dim=4,
            hidden_dims=(8, 4),
            domain_aware_indices=(0, 1),
        ),
    )
    output = model(
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        torch.randn(3, 1),
        torch.tensor([0, 1, 2]),
    )
    assert output.auxiliary_loss.ndim == 0
    assert "mean_sparsity" in output.diagnostics
    assert 0.0 <= float(output.diagnostics["mean_sparsity"]) <= 1.0


def test_adl_checkpoint_contains_centers():
    model = build_model("adl", ModelConfig([5], 0, 2, embedding_dim=3, hidden_dims=(4, 3), cluster_num=2))
    assert "router.centers" in model.state_dict()


def test_factorized_layer_grouped_computation_matches_direct_weights():
    layer = FactorizedClusterLayer(input_dim=3, output_dim=2, cluster_num=2)
    inputs = torch.randn(5, 3, requires_grad=True)
    route = torch.tensor([0, 1, 1, 0, 1])
    actual = layer(inputs, route)
    selected = layer.cluster_weight[route] * layer.shared_weight.unsqueeze(0)
    expected = torch.bmm(selected, inputs.unsqueeze(2)).squeeze(2)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    assert inputs.grad is not None


def test_factorized_layer_uses_autocast_output_dtype():
    layer = FactorizedClusterLayer(input_dim=4, output_dim=3, cluster_num=2)
    inputs = torch.randn(6, 4)
    route = torch.tensor([0, 1, 1, 0, 1, 0])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = layer(inputs, route)
    assert output.dtype == torch.bfloat16
    assert output.shape == (6, 3)
