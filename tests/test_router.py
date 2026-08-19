import torch
import torch.nn.functional as F

from adl_repro.models import DLMRouter


def test_router_applies_ewma_once_after_iterations():
    router = DLMRouter(input_dim=2, cluster_num=2, beta=0.5, iterations=3)
    initial = F.normalize(torch.tensor([[1.0, 0.2], [-0.4, 1.0]]), dim=1)
    router.centers.copy_(initial)
    points = torch.tensor([[2.0, 0.0], [1.0, 0.2], [0.0, 2.0], [0.1, 1.0]])

    current = initial.clone()
    for _ in range(3):
        coefficient = (points @ current.T).softmax(dim=1)
        current = F.normalize(coefficient.T @ points, dim=1)
    expected = F.normalize(0.5 * initial + 0.5 * current, dim=1)

    router.train()
    router(points)
    assert torch.allclose(router.centers, expected, atol=1e-6)


def test_eval_does_not_update_centers():
    router = DLMRouter(input_dim=3, cluster_num=2)
    before = router.centers.clone()
    router.eval()
    router(torch.randn(8, 3))
    assert torch.equal(before, router.centers)

