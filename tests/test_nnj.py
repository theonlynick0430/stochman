import numpy
import pytest
import torch

from stochman import nnj


def _fd_jacobian(function, x, h=1e-4):
    """Compute finite difference Jacobian of given function
    at a single location x. This function is mainly considered
    useful for debugging."""

    no_batch = x.dim() == 1
    if no_batch:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise Exception("The input should be a D-vector or a BxD matrix")
    B, D = x.shape

    # Compute finite differences
    E = h * torch.eye(D)
    try:
        # Disable "training" in the function (relevant eg. for batch normalization)
        orig_state = function.disable_training()
        Jnum = torch.cat([((function(x[b] + E) - function(x[b].unsqueeze(0))).t() / h).unsqueeze(0) for b in range(B)])
    finally:
        function.enable_training(orig_state)  # re-enable training

    if no_batch:
        Jnum = Jnum.squeeze(0)

    return Jnum


def _jacobian_check(function, in_dim=None):
    """Accepts an nnj module and checks the
    Jacobian via the finite differences method.

    Args:
        function:   An nnj module object. The
                    function to be tested.

    Returns a tuple of the following form:
    (Jacobian_analytical, Jacobian_finite_differences)
    """

    with torch.no_grad():
        batch_size = 5
        if in_dim is None:
            in_dim, _ = function.dimensions()
            if in_dim is None:
                in_dim = 10
        x = torch.randn(batch_size, in_dim)
        try:
            orig_state = function.disable_training()
            y, J, Jtype = function(x, jacobian=True, return_jac_type=True)
        finally:
            function.enable_training(orig_state)

        if Jtype is nnj.JacType.DIAG:
            J = J.diag_embed()

        Jnum = _fd_jacobian(function, x)

        return J, Jnum


_in_features = 10
_models = [
    nnj.Sequential(nnj.Linear(_in_features, 2), nnj.Softplus(beta=100, threshold=5), nnj.Linear(2, 4), nnj.Tanh()),
    nnj.Sequential(nnj.RBF(_in_features, 30), nnj.Linear(30, 2)),
    nnj.Sequential(nnj.Linear(_in_features, 4), nnj.Norm2()),
    nnj.Sequential(nnj.Linear(_in_features, 50), nnj.ReLU(), nnj.Linear(50, 100), nnj.Softplus()),
    nnj.Sequential(nnj.Linear(_in_features, 256)),
    nnj.Sequential(nnj.Softplus(), nnj.Linear(_in_features, 3), nnj.Softplus()),
    nnj.Sequential(nnj.Softplus(), nnj.Sigmoid(), nnj.Linear(_in_features, 3)),
    nnj.Sequential(nnj.Softplus(), nnj.Sigmoid()),
    nnj.Sequential(nnj.Linear(_in_features, 3), nnj.OneMinusX()),
    nnj.Sequential(
        nnj.PosLinear(_in_features, 2), nnj.Softplus(beta=100, threshold=5), nnj.PosLinear(2, 4), nnj.Tanh()
    ),
    nnj.Sequential(nnj.PosLinear(_in_features, 5), nnj.Reciprocal(b=1.0)),
    nnj.Sequential(nnj.ReLU(), nnj.ELU(), nnj.LeakyReLU(), nnj.Sigmoid(), nnj.Softplus(), nnj.Tanh()),
    nnj.Sequential(nnj.ReLU()),
    nnj.Sequential(nnj.ELU()),
    nnj.Sequential(nnj.LeakyReLU()),
    nnj.Sequential(nnj.Sigmoid()),
    nnj.Sequential(nnj.Softplus()),
    nnj.Sequential(nnj.Tanh()),
    nnj.Sequential(nnj.Hardshrink()),
    nnj.Sequential(nnj.Hardtanh()),
    nnj.Sequential(nnj.ResidualBlock(nnj.Linear(_in_features, 50), nnj.ReLU())),
    nnj.Sequential(nnj.BatchNorm1d(_in_features)),
    nnj.Sequential(
        nnj.BatchNorm1d(_in_features),
        nnj.ResidualBlock(nnj.Linear(_in_features, 25), nnj.Softplus()),
        nnj.BatchNorm1d(25),
        nnj.ResidualBlock(nnj.Linear(25, 25), nnj.Softplus()),
    ),
]


@pytest.mark.parametrize("model", _models)
def test_jacobians(model):
    """Test that the analytical jacobian of the model is consistent with finite
    order approximation
    """
    J, Jnum = _jacobian_check(model, _in_features)
    numpy.testing.assert_allclose(J, Jnum, rtol=1, atol=1e-2)
