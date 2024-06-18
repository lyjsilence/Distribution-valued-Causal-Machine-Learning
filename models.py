import numpy as np
import torch
import copy
import torch.nn as nn
from scipy.interpolate import BSpline
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions.normal import Normal
from MovingBatchNorm1d import MovingBatchNorm1d
from baseline_models import est_ECDF
import utils

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "sigmoid": nn.Sigmoid(),
    "softsign": nn.Softsign(),
    "selu": nn.SELU(),
    "softmax": nn.Softmax(dim=1)
}


class LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden = hidden
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(hidden))
        self.beta = nn.Parameter(torch.randn(hidden))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x-mean) / std * self.alpha + self.beta


# Numerical MLP
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=[64, 64, 64], activation='relu', dropout=0.1, device='cpu'):
        super(MLP, self).__init__()
        self.activation = ACTIVATIONS[activation]
        self.dim = [dim_in] + dim_hidden + [dim_out]
        self.linears = nn.ModuleList([nn.Linear(self.dim[i-1], self.dim[i]) for i in range(1, len(self.dim))])
        self.layernorms = nn.ModuleList([LayerNorm(hidden) for hidden in dim_hidden])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(dim_hidden))])

    def forward(self, x):
        for i in range(len(self.dim)-2):
            x = self.linears[i](x)
            x = x + self.layernorms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        x = self.linears[-1](x)
        return x

class simple_MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=[64, 64, 64], activation='relu', dropout=0.1, device='cpu'):
        super(simple_MLP, self).__init__()
        self.activation = ACTIVATIONS[activation]
        self.dim = [dim_in] + dim_hidden + [dim_out]
        self.linears = nn.ModuleList([nn.Linear(self.dim[i-1], self.dim[i]) for i in range(1, len(self.dim))])
        self.layernorms = nn.ModuleList([LayerNorm(hidden) for hidden in dim_hidden])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(dim_hidden))])

    def forward(self, x):
        for i in range(len(self.dim)-2):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        x = self.linears[-1](x)
        return x

# Functional_MLP
class Functional_MLP(nn.Module):
    def __init__(self, dim_out, t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], device='cpu'):
        super(Functional_MLP, self).__init__()
        self.dim_out = dim_out
        self.device = device

        spline_deg = 3
        self.num_basis = len(t) - spline_deg - 1
        self.basis_layer = []
        for i in range(self.num_basis):
            const_basis = np.zeros(self.num_basis)
            const_basis[i] = 1.0
            self.basis_layer.append(BSpline(np.array(t), const_basis, spline_deg))
        # the weights for each basis
        self.alpha = nn.Parameter(torch.randn(self.dim_out, self.num_basis))
        self.beta = nn.Parameter(torch.randn(self.num_basis))
        torch.nn.init.xavier_uniform_(self.alpha)

        self.t = torch.tensor(t).to(self.device)

    def forward(self, x):
        t = self.t.unsqueeze(1).cpu().detach().numpy()
        self.bases = [torch.tensor(basis(t).transpose(-1, -2)).to(torch.float32).to(self.device) for basis in self.basis_layer]
        y = 0
        for j in range(x.shape[1]):
            betas = torch.sum(torch.cat([self.alpha[j][k] * self.bases[k] for k in range(self.num_basis)]), dim=0, keepdim=True)
            y += x[:, j].unsqueeze(1).repeat([1, betas.shape[1]]) * betas
        return y


# Neural Functional Regression for arbitrary quantile
class NFR(nn.Module):
    def __init__(self, dim_in, dim_out, t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dim_hidden=[64, 64, 64],
                 activation='relu', dropout=0.1, device='cpu'):
        super(NFR, self).__init__()
        '''
        dim_in: dimension of features
        dim_out: dimension of output from numerical MLP
        num_basis: number of basis used
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device
        self.activation = ACTIVATIONS[activation]
        self.dropout = dropout

        # Numerical MLP for scalar input and representation learning
        self.MLP = MLP(dim_in, dim_out, dim_hidden, activation=activation, dropout=self.dropout)

        # Functional MLP for scalar input and functional output
        self.Functional_MLP = Functional_MLP(dim_out, t, device=device)


    def forward(self, X):
        X = self.MLP(X)
        X = self.Functional_MLP(X)
        return X


# Neural Functional Regression with representation learning for arbitrary quantile
class rep_NFR(nn.Module):
    def __init__(self, dim_in, dim_out, t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dim_hidden_rep=[64, 64, 64],
                 dim_hidden_head=[32, 32], num_treatment=5, activation='relu', dropout=0.1, device='cpu'):
        super(rep_NFR, self).__init__()
        '''
        dim_in: dimension of features
        dim_out: dimension of output from numerical MLP
        num_basis: number of basis used
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_treatment = num_treatment
        self.t = t
        self.device = device
        self.activation = ACTIVATIONS[activation]
        self.dropout = dropout

        # Numerical MLP for scalar input and representation learning
        self.MLP = MLP(dim_in, dim_out, dim_hidden_rep, activation=activation, dropout=self.dropout)

        # headers
        headers = []
        for i in range(self.num_treatment):
            headers.append(nn.Sequential(
                               MLP(dim_out, dim_in, dim_hidden_head, activation=activation, dropout=self.dropout),
                               Functional_MLP(dim_in, t, device=device)
                           ))
        self.headers = nn.ModuleList(headers)

    def forward(self, X):
        X, D = X[:, :-1], X[:, -1]
        X = self.MLP(X)
        y = torch.zeros([X.shape[0], len(self.t)]).to(self.device)
        for i in range(self.num_treatment):
            y += self.headers[i](X) * ((D == i) * 1.0).unsqueeze(1)
        return y


# Normalizing Flow
class Gated_Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Gated_Linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.bias = nn.Linear(1, dim_out, bias=False)
        self.gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        gate = torch.sigmoid(self.gate(t))
        return self.linear(x) * gate + self.bias(t)


class ODENet(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, hidden_layer=2):
        super(ODENet, self).__init__()
        # input layer
        linear_layers = [Gated_Linear(input_dim+cond_dim, hidden_dim)]
        act_layers = [nn.Softplus()]
        # hidden layer
        for i in range(hidden_layer):
            linear_layers.append(Gated_Linear(hidden_dim, hidden_dim))
            act_layers.append(nn.Softplus())
        # output layer
        linear_layers.append(Gated_Linear(hidden_dim, input_dim))

        self.linear_layers = nn.ModuleList(linear_layers)
        self.act_layers = nn.ModuleList(act_layers)

    def forward(self, t, x, cond):
        dx = torch.cat([x, cond], dim=1)
        for l, layer in enumerate(self.linear_layers):
            dx = layer(t, dx)
            if l < len(self.linear_layers) - 1:
                dx = self.act_layers[l](dx)
        return dx


class ODEFunc(nn.Module):
    def __init__(self, ODENet, rademacher=False, div_samples=1):
        super(ODEFunc, self).__init__()

        self.ODENet = ODENet
        self.rademacher = rademacher
        self.div_samples = div_samples
        self.divergence_fn = self.divergence_approx

    def divergence_approx(self, f, z, e=None):
        samples = []
        sqnorms = []
        for e_ in e:
            e_dzdx = torch.autograd.grad(f, z, e_, create_graph=True)[0]
            n = e_dzdx.view(z.size(0), -1).pow(2).mean(dim=1, keepdim=True)
            sqnorms.append(n)
            e_dzdx_e = e_dzdx * e_
            samples.append(e_dzdx_e.view(z.shape[0], -1).sum(dim=1, keepdim=True))

        S = torch.cat(samples, dim=1)
        approx_tr_dzdx = S.mean(dim=1)
        N = torch.cat(sqnorms, dim=1).mean(dim=1)

        return approx_tr_dzdx, N

    def sample_rademacher(self, y):
        return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

    def sample_gaussian(self, y):
        return torch.randn_like(y)

    def before_odeint(self, e=None):
        self._e = e
        self._sqjacnorm = None

    def forward(self, t, states):
        assert len(states) >= 2
        z = states[0]
        cond = states[2]

        if len(z.shape) == 1:
            z = z.unsqueeze(1)

        # convert to tensor
        t = t.to(z)

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = [self.sample_rademacher(z) for k in range(self.div_samples)]
            else:
                self._e = [self.sample_gaussian(z) for k in range(self.div_samples)]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t = torch.ones(z.size(0), 1).to(z) * t.clone().detach().requires_grad_(True).type_as(z)

            for t_ in states[3:]:
                t_.requires_grad_(True)

            # compute dz by ODEnet
            dz = self.ODENet(t, z, cond, *states[3:])

            # Compute tr(df/dz)
            if not self.training and dz.view(dz.shape[0], -1).shape[1] == 2:
                divergence = self.divergence_bf(dz, z).view(z.shape[0], 1)
            else:
                divergence, sqjacnorm = self.divergence_fn(dz, z, e=self._e)
                divergence = divergence.view(z.shape[0], 1)
            self.sqjacnorm = sqjacnorm

        return tuple([dz, -divergence, torch.zeros_like(cond).requires_grad_(True)] +
                     [torch.zeros_like(s_).requires_grad_(True) for s_ in states[3:]])


class RegularizedODEfunc(nn.Module):
    def __init__(self, ODEFunc, reg_fns):
        super(RegularizedODEfunc, self).__init__()
        self.ODEFunc = ODEFunc
        self.reg_fns = reg_fns

    def before_odeint(self, *args, **kwargs):
        self.ODEFunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):

        with torch.enable_grad():
            x, logp, cond = state[:3]
            x.requires_grad_(True)
            t.requires_grad_(True)
            logp.requires_grad_(True)
            dstate = self.ODEFunc(t, (x, logp, cond))
            if len(state) > 3:
                dx, dlogp, cond = dstate[:3]
                reg_states = tuple(reg_fn(x, t, logp, dx, dlogp, self.ODEFunc) for reg_fn in self.reg_fns)
                return dstate + reg_states
            else:
                return dstate


class CNF(nn.Module):
    def __init__(self, ODEFunc, T=1.0, reg_fns=None, train_T=False, solver='dopri5', atol=1e-6, rtol=1e-4):
        super(CNF, self).__init__()
        if train_T:
            self.sqrt_end_time = nn.Parameter(torch.sqrt(torch.tensor(T)), requires_grad=True)
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if reg_fns is not None:
            ODEFunc = RegularizedODEfunc(ODEFunc, reg_fns)
            nreg = len(reg_fns)

        self.nreg = nreg
        self.ODEFunc = ODEFunc
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def _flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, z_, logpz=None, reg_states=tuple(), integration_times=None, reverse=False):

        # separate z and conditional variables
        z = z_[:, 0]
        cond = z_[:, 1:]

        if not len(reg_states) == self.nreg and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = self._flip(integration_times, 0)

        # Refresh
        self.ODEFunc.before_odeint()

        if self.training:
            state_t = odeint(self.ODEFunc, (z, _logpz, cond) + reg_states, integration_times.to(z),
                             atol=self.atol, rtol=self.rtol, method=self.solver, options=self.solver_options)
        else:
            state_t = odeint(self.ODEFunc, (z, _logpz, cond), integration_times.to(z),
                             atol=self.test_atol, rtol=self.test_rtol, method=self.test_solver)

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t, cond = state_t[:3]
        reg_states = state_t[3:]

        if logpz is not None:
            return z_t, logpz_t, reg_states
        else:
            return z_t


class Cond_CNF(nn.Module):
    def __init__(self, args, device):
        super(Cond_CNF, self).__init__()

        reg_fns, reg_coeffs = utils.creat_reg_fns(args)

        chain = [self.build_cnf(args, reg_fns) for _ in range(args.num_blocks)]

        if args.batch_norm:
            bn_dim = args.input_dim
            bn_layers = [MovingBatchNorm1d(bn_dim, bn_lag=0) for _ in range(args.num_blocks)]
            bn_chain = [MovingBatchNorm1d(bn_dim, bn_lag=0)]
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
            chain = bn_chain

        self.reg_coeffs = reg_coeffs
        self.chain = nn.ModuleList(chain)

    def build_cnf(self, args, reg_fns):
        f = ODENet(input_dim=args.input_dim, cond_dim=args.cond_dim, hidden_dim=args.hidden_dim)
        f_aug = ODEFunc(ODENet=f, rademacher=args.rademacher)
        cnf = CNF(ODEFunc=f_aug, T=args.flow_time, reg_fns=reg_fns, train_T=args.train_T, solver=args.solver)
        return cnf

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        for i in inds:
            if isinstance(self.chain[i], MovingBatchNorm1d):
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            else:
                x, logpx, reg_states = self.chain[i](x, logpx, reverse=reverse)

        if len(x.shape) == 1:
            x.unsqueeze(1)
        return x, logpx, reg_states, self.reg_coeffs

class Flow_IPW(nn.Module):
    def __init__(self, args, device):
        super(Flow_IPW, self).__init__()
        self.flow = Cond_CNF(args, device)
        self.input_size = args.num_cov
        self.hidden_size = args.hidden_dim
        self.num_obs = args.num_obs
        self.softplus = nn.Softplus()
        self.device = device

        self.norm_fn = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

    def compute_loss(self, z, delta_logp, params, reg_states, reg_coeffs, test=False):
        mean, var = torch.chunk(params, 2, dim=1)
        var = self.softplus(var)

        logpz = torch.sum(self.base_dist(mean, var).log_prob(z), dim=1, keepdim=True)
        logpx = logpz - delta_logp

        if test:
            return logpx
        else:
            nll = -torch.mean(logpx)

            reg_states = tuple(torch.mean(rs) for rs in reg_states)
            if reg_coeffs:
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs) if coeff != 0
                )
            return nll, reg_loss

    def base_dist(self, mean, var):
        return Normal(mean, var)

    def forward(self, X, D, test=False):
        if len(D.shape) == 1:
            D = D.unsqueeze(1)
        params = self.norm_fn(X)
        DX = torch.cat([D, X], dim=1)
        logpy = torch.zeros(DX.shape[0], 1).to(DX)
        z, delta_logp, reg_states, reg_coeffs = self.flow(DX, logpy)
        if test:
            logpx = self.compute_loss(z, delta_logp, params, reg_states, reg_coeffs, test=test)
            return torch.exp(logpx)
        else:
            loss_nll, loss_reg = self.compute_loss(z, delta_logp, params, reg_states, reg_coeffs, test=test)
            return loss_nll, loss_reg
