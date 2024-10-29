#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from matplotlib.axis import Axis
from torch import nn


class BasicCurve(ABC, nn.Module):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Abstract class that represents a batch of curves in space. 
        Curves are parametrized using t in [0, 1].

        Args: 
            begin (torch.Tensor): start points of shape [B, D] or [D]
            end (torch.Tensor): end points of shape [B, D] or [D]
            num_nodes (int): number of nodes used for approximation
            requires_grad (bool): if True, compute gradients for curve parameters
            args: arguments specific to curve implementation
            kwargs: keyword arguments specific to curve implementation
        """
        super().__init__()
        self._num_nodes = num_nodes
        self._requires_grad = requires_grad

        # if either begin or end only has one point, while the other has a batch
        # then we expand the singular point. End result is that both begin and
        # end should have shape BxD
        batch_begin = 1 if len(begin.shape) == 1 else begin.shape[0]
        batch_end = 1 if len(end.shape) == 1 else end.shape[0]
        if batch_begin == 1 and batch_end == 1:
            _begin = begin.detach().view((1, -1))  # 1xD
            _end = end.detach().view((1, -1))  # 1xD
        elif batch_begin == 1:  # batch_end > 1
            _begin = begin.detach().view((1, -1)).repeat(batch_end, 1)  # BxD
            _end = end.detach()  # BxD
        elif batch_end == 1:  # batch_begin > 1
            _begin = begin.detach()  # BxD
            _end = end.detach().view((1, -1)).repeat(batch_begin, 1)  # BxD
        elif batch_begin == batch_end:
            _begin = begin.detach()  # BxD
            _end = end.detach()  # BxD
        else:
            raise ValueError("BasicCurve.__init__ requires begin and end points to have " "the same shape")

        # register begin and end as buffers
        self.register_buffer("begin", _begin)  # BxD
        self.register_buffer("end", _end)  # BxD

        # overriden by child modules
        self._init_params(*args, **kwargs)

    @abstractmethod
    def _init_params(self, *args, **kwargs) -> None:
        pass

    @property
    def device(self):
        """Returns the device of the curves."""
        return self.params.device

    def __len__(self):
        """Returns the batch dimension e.g. the number of curves."""
        return self.begin.shape[0]

    def plot(
        self, t0: float = 0.0, t1: float = 1.0, N: int = 100, ax: Axis = None, *plot_args, **plot_kwargs
    ):
        """
        Plot each curve between @t0 and @t1.

        Args:
            t0 (float): start time
            t1 (float): end time
            N (int): number of points used for plotting curves
            ax (Axis): (optional) object to specify where curves should be plotted. If None, uses
                matplotlib.pyplot. Defaults to None. 
            plot_args: additional arguments passed directly to plt.plot
            plot_kwargs: additional keyword-arguments passed directly to plt.plot

        Returns:
            figs (array): array of size [B] of figure handles
        """
        with torch.no_grad():
            import matplotlib.pyplot as plt

            t = torch.linspace(t0, t1, N, dtype=self.begin.dtype, device=self.device)
            points = self(t)  # NxD or BxNxD

            if len(points.shape) == 2:
                points.unsqueeze_(0)  # 1xNxD

            plot_in = ax or plt
            if ax is not None:
                t = t.detach().numpy()
                points = points.detach().numpy()

            figs = []
            if points.shape[-1] == 1:
                for b in range(points.shape[0]):
                    fig = plot_in.plot(t, points[b], *plot_args, **plot_kwargs)
                    figs.append(fig)
                return figs
            if points.shape[-1] == 2:
                for b in range(points.shape[0]):
                    fig = plot_in.plot(points[b, :, 0], points[b, :, 1], *plot_args, **plot_kwargs)
                    figs.append(fig)
                return figs

            raise ValueError(
                "BasicCurve.plot only supports plotting curves in"
                f" 1D or 2D, but recieved points with shape {points.shape}"
            )

    def euclidean_length(self, t0: float = 0.0, t1: float = 1.0, N: int = 100) -> torch.Tensor:
        """
        Calculate the euclidian length of each curve between @t0 and @t1.

        Args:
            t0 (float): start time
            t1 (float): end time
            N (int): number of discretized points

        Returns:
            lengths (torch.Tensor): tensor of shape [B] with the length of each curve
        """
        t = torch.linspace(t0, t1, N, device=self.device)  # N
        points = self(t)  # NxD or BxNxD
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # 1xNxD
        delta = points[:, 1:] - points[:, :-1]  # Bx(N-1)xD
        energies = (delta**2).sum(dim=2)  # Bx(N-1)
        lengths = energies.sqrt().sum(dim=1)  # B
        return lengths

    def fit(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        num_steps: int = 50,
        threshold: float = 1e-6,
        **optimizer_kwargs,
    ) -> torch.Tensor:
        """
        Fit each curve by minimizing |@x - self(@t)|².

        Args:
            t (torch.tensor): tensor of shape [B, N] or [N] with times to evaluate each curve at
            x (torch.tensor): tensor of shape [B, N, D] or [N, D] containing values each curve 
                should take at times in @t.
            num_steps (int): number of optimization steps
            threshold (float): stopping criterium
            optimizer_kwargs: additional keyword arguments (like lr) passed to the optimizer

        Returns:
            loss: optimized loss
        """
        # using a second order method on a linear problem should imply
        # that we get to the optimum in few iterations (ideally 1).
        opt = torch.optim.LBFGS(self.parameters(), **optimizer_kwargs)
        loss_func = torch.nn.MSELoss()

        def closure():
            opt.zero_grad()
            L = loss_func(self(t), x)
            L.backward()
            return L

        with torch.enable_grad():
            for _ in range(num_steps):
                loss = opt.step(closure=closure)
                if torch.max(torch.abs(self.params.grad)) < threshold:
                    break
        return loss


class DiscreteCurve(BasicCurve):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        params: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Class that represents a batch of discrete curves in space. 
        In particular, we approximate a curve by linearly interpolating between 
        @num_nodes points. Curves are parametrized using t in [0, 1].

        Args: 
            begin (torch.Tensor): start points of shape [B, D] or [D]
            end (torch.Tensor): end points of shape [B, D] or [D]
            num_nodes (int): number of nodes used for approximation
            requires_grad (bool): if True, compute gradients for curve parameters
            params (torch.Tensor): (optional) curve parameters of shape [B, num_nodes-2, D]. 
                If None, initializes params to linearly interpolate between @begin and @end.
                Defaults to None.
        """
        super().__init__(begin, end, num_nodes, requires_grad, params=params)

    def _init_params(self, params, *args, **kwargs) -> None:
        self.register_buffer(
            "t",
            torch.linspace(0, 1, self._num_nodes, dtype=self.begin.dtype)[1:-1] # exclude endpoints
            .view(1, -1, 1)
            .expand(self.begin.shape[0], -1, self.begin.shape[1]),  # Bx(num_nodes-2)xD
        )
        if params is None:
            params = self.t * self.end.unsqueeze(1) + (1 - self.t) * self.begin.unsqueeze(
                1
            )  # Bx(num_nodes)xD
        if self._requires_grad:
            self.register_parameter("params", nn.Parameter(params))
        else:
            self.register_buffer("params", params)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate each curve at times in @t.

        Args: 
            t (torch.Tensor): tensor of shape [B, N] or [N] with times to evaluate each curve at 

        Returns: 
            result (torch.Tensor): tensor of shape [B, N] or [N] with values of each curve 
                at requested times

        Note: each t must be in [0,1]
        """
        start_nodes = torch.cat((self.begin.unsqueeze(1), self.params), dim=1)  # Bx(num_edges)xD
        end_nodes = torch.cat((self.params, self.end.unsqueeze(1)), dim=1)  # Bx(num_edges)xD
        B, num_edges, D = start_nodes.shape
        t0 = torch.cat(
            (
                torch.zeros(B, 1, D, dtype=self.t.dtype, device=self.device),
                self.t,
                torch.ones(B, 1, D, dtype=self.t.dtype, device=self.device),
            ),
            dim=1,
        )  # Bx(num_nodes)xD
        a = (end_nodes - start_nodes) / (t0[:, 1:] - t0[:, :-1])  # Bx(num_edges)xD
        b = start_nodes - a * t0[:, :-1]  # Bx(num_edges)xD

        if t.ndim == 1:
            tt = t.view((1, -1)).expand(B, -1)  # Bx|t|
        elif t.ndim == 2:
            tt = t  # Bx|t|
        else:
            raise Exception("t must have at most 2 dimensions")
        idx = (
            (torch.floor(tt * num_edges).clamp(min=0, max=num_edges - 1).long())  # Bx|t|
            .unsqueeze(2)
            .repeat(1, 1, D)
        ).to(
            self.device
        )  # Bx|t|xD, this assumes that nodes are equi-distant
        result = torch.gather(a, 1, idx) * tt.unsqueeze(2) + torch.gather(b, 1, idx)  # Bx|t|xD
        if B == 1:
            result = result.squeeze(0)  # |t|xD
        return result

    def __getitem__(self, indices: int) -> "DiscreteCurve":
        params = self.params[indices]
        if params.dim() == 2:
            params = params.unsqueeze(0)
        C = DiscreteCurve(
            begin=self.begin[indices],
            end=self.end[indices],
            num_nodes=self._num_nodes,
            requires_grad=self._requires_grad,
            params=params,
        ).to(self.device)
        return C

    def __setitem__(self, indices, curves) -> None:
        self.params[indices].data = curves.params.squeeze()

    def constant_speed(
        self, metric=None, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparametrize each curve to have constant speed.

        Args:
            metric (Manifold): (optional) Manifold under which the curve should have constant speed.
                If None, then the Euclidean metric is applied. Defaults to None.
            t (torch.Tensor): (optional) tensor of shape [B, N] or [N] of times to query each curve at. 
                If None, use 100 equally spaced times in [0, 1]. Defaults to None.

        Note: It is not possible to back-propagate through this function.
        """
        from stochman import CubicSpline

        with torch.no_grad():
            if t is None:
                t = torch.linspace(0, 1, 100)  # N
            Ct = self(t)  # NxD or BxNxD
            if Ct.ndim == 2:
                Ct.unsqueeze_(0)  # BxNxD
            B, N, D = Ct.shape
            delta = Ct[:, 1:] - Ct[:, :-1]  # Bx(N-1)xD
            if metric is None:
                local_len = delta.norm(dim=2)  # Bx(N-1)
            else:
                local_len = (
                    metric.inner(Ct[:, :-1].reshape(-1, D), delta.view(-1, D), delta.view(-1, D))
                    .view(B, N - 1)
                    .sqrt()
                )  # Bx(N-1)
            cs = local_len.cumsum(dim=1)  # Bx(N-1)
            zero = torch.zeros(B, 1, dtype=cs.dtype, device=cs.device)  # Bx1
            one = torch.ones(B, 1, dtype=cs.dtype, device=cs.device)  # Bx1
            new_t = torch.cat((zero, cs / cs[:, -1].unsqueeze(1)), dim=1)  # BxN
            S = CubicSpline(zero, one)
            _ = S.fit(new_t, t.unsqueeze(0).expand(B, -1).unsqueeze(2))
            new_params = self(S(self.t[:, :, 0]).squeeze(-1))  # Bx(num_nodes-2)xD
            self.params = nn.Parameter(new_params)
            return new_t, Ct, local_len.sum(dim=1)

    def tospline(self):
        """Returns discrete curve converted to cubic spline."""
        from stochman import CubicSpline

        c = CubicSpline(
            begin=self.begin,
            end=self.end,
            num_nodes=self._num_nodes,
            requires_grad=self._requires_grad,
        )
        _ = c.fit(self.t[0, :, 0], self.params)
        return c


class CubicSpline(BasicCurve):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        basis: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Class that approximates a batch of curves in space using cubic splines.
        In particular, we approximate a curve using @num_nodes-1 individual cubic splines
        and compute a basis for the coefficients of the cubic splines that enforce 
        necessary constraints (see https://www.youtube.com/watch?v=wMMjF7kXnWA). 
        Curves are parametrized using t in [0, 1].

        Args: 
            begin (torch.Tensor): start points of shape [B, D] or [D]
            end (torch.Tensor): end points of shape [B, D] or [D]
            num_nodes (int): number of nodes used for approximation
            requires_grad (bool): if True, compute gradients for curve parameters
            basis (torch.Tensor): (optional) tensor of shape [B, D, 4*(num_nodes-1), K] or 
                [D, 4*(num_nodes-1), K] that consists of K basis vectors for the coefficients 
                of the cubic splines for each batch of curves. If None, the basis will be 
                computed using self._compute_basis(). Defaults to None.
            params (torch.Tensor): (optional) tensor of shape [B, D, K] or [D, K] of 
                parameters that specify linear combinations of @basis. If None, 
                params will be initialized to zero. Defaults to None.
        """
        super().__init__(begin, end, num_nodes, requires_grad, basis=basis, params=params)

    def _init_params(self, basis, params) -> None:
        pass
        if basis is None:
            basis = self._compute_basis(num_edges=self._num_nodes - 1).to(self.begin.device)
        self.register_buffer("basis", basis)

        if params is None:
            # must fit splines for each dimension provided 
            # ex: for a 2D curve we must fit splines for x(t) and y(t)
            params = torch.zeros(
                self.begin.shape[0], self.basis.shape[1], self.begin.shape[1],
                dtype=self.begin.dtype, device=self.begin.device
            ) # shape: Bx[dim(basis)]xD
        else:
            params = params.unsqueeze(0) if params.ndim == 2 else params

        if self._requires_grad:
            self.register_parameter("params", nn.Parameter(params))
        else:
            self.register_buffer("params", params)

    # Note: constraints are imposed at times that are independent of points we are fitting curve to
    def _compute_basis(self, num_edges, thresh=1e-5) -> torch.Tensor:
        with torch.no_grad():
            B = self.begin.shape[0] # batch dim
            D = self.begin.shape[1] # space dim
            num_coeff = 4 * num_edges

            # fix boundary points
            boundary_points = torch.zeros(B, D, 2, num_coeff + 1, dtype=self.begin.dtype)
            boundary_points[:, :, 0, 0] = 1.0
            boundary_points[:, :, 0, -1] = self.begin
            boundary_points[:, :, 1, -5:-1] = 1.0
            boundary_points[:, :, 1, -1] = self.end

            # natural boundary conditions: S"(0)=S"(1)=0
            natural_boundary = torch.zeros(B, D, 2, num_coeff + 1, dtype=self.begin.dtype)
            natural_boundary[:, :, 0, 2] = 2.0
            natural_boundary[:, :, 1, -5:-1] = torch.tensor([0.0, 0.0, 2.0, 6.0])

            # no need to shift constraints since a + b(t-t0) + c(t-t0)^2 + d(t-t0)^3 
            # can be reformatted to e + f*t + g*t^2 + h*t^3
            t = torch.linspace(0, 1, num_edges + 1, dtype=self.begin.dtype)[1:-1] # exclude end points

            # zeroth derivative conditions: S_i(t_i) = S_(i+1)(t_i)
            zeroth = torch.zeros(B, D, num_edges - 1, num_coeff + 1, dtype=self.begin.dtype)
            for i in range(num_edges-1):
                si = 4 * i
                fill = torch.tensor([1.0, t[i], t[i] ** 2, t[i] ** 3], dtype=self.begin.dtype)
                zeroth[:, :, i, si:(si + 4)] = fill
                zeroth[:, :, i, (si + 4):(si + 8)] = -fill

            # first derivative conditions: S_i'(t_i) = S_(i+1)'(t_i)
            first = torch.zeros(B, D, num_edges - 1, num_coeff + 1, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i
                fill = torch.tensor([0.0, 1.0, 2.0 * t[i], 3.0 * t[i] ** 2], dtype=self.begin.dtype)
                first[:, :, i, si:(si + 4)] = fill
                first[:, :, i, (si + 4):(si + 8)] = -fill

            # second derivative conditions: S_i"(t_i) = S_(i+1)"(t_i)
            second = torch.zeros(B, D, num_edges - 1, num_coeff + 1, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i
                fill = torch.tensor([0.0, 0.0, 2.0, 6.0 * t[i]], dtype=self.begin.dtype)
                second[:, :, i, si:(si + 4)] = fill
                second[:, :, i, (si + 4):(si + 8)] = -fill

            # represents under-constrained system of eqns. for coefficients of cubic splines between each node
            constraints = torch.cat((boundary_points, natural_boundary, zeroth, first, second), dim=2)
            self.constraints = constraints # shape: [B, D, -1, num_coeffs + 1]

            # solution to system of eqns. is the particular solution + nullsapce
            A = constraints[:, :, :, :-1].view(B * D, -1, 4 * num_edges)
            b = constraints[:, :, :, -1].view(B * D, -1)
            x, _, _, _ = torch.linalg.lstsq(A, b)
            _, S, V = torch.svd(A, some=False)
            nullspace = V[:, :, S.shape[1]:]  
            basis = torch.cat((nullspace, x.view(B*D, -1, 1)), dim=2).view(B, D, 4 * num_edges, -1)

            return basis # shape: [B, D, num_coeffs, dim(nullspace)+1=dim(basis)]

    def _get_coeffs(self) -> torch.Tensor:
        coeffs = (
            self.basis.unsqueeze(0).expand(self.params.shape[0], -1, -1).bmm(self.params)
        )  # Bx[num_coeffs]xD
        B, num_coeffs, D = coeffs.shape
        degree = 4
        num_edges = num_coeffs // degree
        coeffs = coeffs.view(B, num_edges, degree, D)  # Bx[num_edges]x4xD
        return coeffs

    def _eval_polynomials(self, t: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        # each row of coeffs should be of the form c0, c1, c2, ... representing polynomials
        # of the form c0 + c1*t + c2*t^2 + ...
        # coeffs: Bx(num_edges)x(degree)xD
        B, num_edges, degree, D = coeffs.shape
        idx = torch.floor(t * num_edges).clamp(min=0, max=num_edges - 1).long()    # B x |t|
        power = (
            torch.arange(0.0, degree, dtype=t.dtype, device=self.device)
            .view(1, 1, -1)
            .expand(B, -1, -1)
        )                                                                           # B x  1  x (degree)
        tpow = t.view(B, -1, 1).pow(power)                                          # B x |t| x (degree)
        coeffs_idx = torch.cat([coeffs[k, idx[k]].unsqueeze(0) for k in range(B)])  # B x |t| x (degree) x D
        retval = tpow.unsqueeze(-1).expand(-1, -1, -1, D) * coeffs_idx              # B x |t| x (degree) x D
        retval = torch.sum(retval, dim=2)                                           # B x |t| x D
        return retval

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate each curve at times in @t.

        Args: 
            t (torch.Tensor): tensor of shape [B, N] or [N] with times to evaluate each curve at 

        Returns: 
            retval (torch.Tensor): tensor of shape [B, N] or [N] with values of each curve 
                at requested times

        Note: each t must be in [0,1]
        """
        coeffs = self._get_coeffs()  # Bx(num_edges)x4xD
        no_batch = t.ndim == 1
        if no_batch:
            t = t.expand(coeffs.shape[0], -1)  # Bx|t|
        retval = self._eval_polynomials(t, coeffs)  # Bx|t|xD
        if no_batch and retval.shape[0] == 1:
            retval.squeeze_(0)  # |t|xD
        return retval

    def __getitem__(self, indices: int) -> "CubicSpline":
        C = CubicSpline(
            begin=self.begin[indices],
            end=self.end[indices],
            num_nodes=self._num_nodes,
            requires_grad=self._requires_grad,
            basis=self.basis,
            params=self.params[indices],
        ).to(self.device)
        return C

    def __setitem__(self, indices, curves) -> None:
        self.params[indices].data = curves.params

    def deriv(self, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate the derivative of each curve.

        Args: 
            t (torch.Tensor): (optional) tensor of shape [B, N] or [N] with times 
                to evaluate the derivative of each curve at. If None, we construct
                the derivative spline. Defaults to None. 

        Returns: 
            retval (CubicSpline/torch.Tensor): if t is None, the derivative spline. 
                Otherwise, tensor of shape [B, N] or [N] with values of the derivative 
                of each curve at requested times.

        Note: each t must be in [0,1]
        """
        coeffs = self._get_coeffs()  # Bx(num_edges)x4xD
        B, num_edges, degree, D = coeffs.shape
        dcoeffs = coeffs[:, :, 1:, :] * torch.arange(
            1.0, degree, dtype=coeffs.dtype, device=self.device
        ).view(1, 1, -1, 1).expand(
            B, num_edges, -1, D
        )  # Bx(num_edges)x3xD
        delta = self.end - self.begin  # BxD
        if t is None:
            # construct the derivative spline
            print("WARNING: Construction of spline derivative objects is currently broken!")
            Z = torch.zeros(B, num_edges, 1, D)  # Bx(num_edges)x1xD
            new_coeffs = torch.cat((dcoeffs, Z), dim=2)  # Bx(num_edges)x4xD
            print("***", new_coeffs[0, 0, :, 0])
            retval = CubicSpline(begin=delta, end=delta, num_nodes=self.num_nodes)
            retval.parameters = (
                retval.basis.t().expand(B, -1, -1).bmm(new_coeffs.view(B, -1, D))
            )  # Bx|parameters|xD
        else:
            if t.dim() == 1:
                t = t.expand(coeffs.shape[0], -1)  # Bx|t|
            # evaluate the derivative spline
            retval = self._eval_polynomials(t, dcoeffs)  # Bx|t|xD
            # tt = t.view((-1, 1)) # |t|x1
            retval += delta.unsqueeze(1)
        return retval

    def constant_speed(
        self, metric=None, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparametrize each curve to have constant speed.

        Args:
            metric (Manifold): (optional) Manifold under which the curve should have constant speed.
                If None, then the Euclidean metric is applied. Defaults to None.
            t (torch.Tensor): (optional) tensor of shape [B, N] or [N] of times to query the curve at. 
                If None, use 100 equally spaced times in [0, 1]. Defaults to None.

        Note: It is not possible to back-propagate through this function.
        """
        with torch.no_grad():
            if t is None:
                t = torch.linspace(0, 1, 100)  # N
            Ct = self(t)  # NxD or BxNxD
            if Ct.dim() == 2:
                Ct.unsqueeze_(0)  # BxNxD
            B, N, D = Ct.shape
            delta = Ct[:, 1:] - Ct[:, :-1]  # Bx(N-1)xD
            if metric is None:
                local_len = delta.norm(dim=2)  # Bx(N-1)
            else:
                local_len = (
                    metric.inner(Ct[:, :-1].reshape(-1, D), delta.view(-1, D), delta.view(-1, D))
                    .view(B, N - 1)
                    .sqrt()
                )  # Bx(N-1)
            cs = local_len.cumsum(dim=1)  # Bx(N-1)
            new_t = torch.cat((torch.zeros(B, 1), cs / cs[:, -1].unsqueeze(1)), dim=1)  # BxN
            _ = self.fit(new_t, Ct)
            return new_t, Ct, local_len.sum(dim=1)

    def todiscrete(self, num_nodes=None):
        """Returns cubic spline converted to discrete curve."""
        from stochman import DiscreteCurve

        if num_nodes is None:
            num_nodes = self._num_nodes
        t = torch.linspace(0, 1, num_nodes)[1:-1]  # (num_nodes-2)
        Ct = self(t)  # Bx(num_nodes-2)xD

        return DiscreteCurve(
            begin=self.begin,
            end=self.end,
            num_nodes=num_nodes,
            requires_grad=self._requires_grad,
            params=Ct,
        )
