from torch.utils.data import Dataset
import torch
import six

class sim_dataset_cdf(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class sim_dataset_reg(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return (self.X[item], self.y[item])


def total_derivative(x, t, logp, dx, dlogp, unused_context):
    del logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1/x.numel(), requires_grad=True)
        tmp = torch.autograd.grad( (u*dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        total_deriv = directional_dx + partial_dt
    except RuntimeError as e:
        if 'One of the differentiated Tensors' in e.__str__():
            raise RuntimeError('No partial derivative with respect to time. Use mathematically equivalent "directional_derivative" regularizer instead')

    tdv2 = total_deriv.pow(2).view(x.size(0), -1)

    return 0.5*tdv2.mean(dim=-1)

def directional_derivative(x, t, logp, dx, dlogp, unused_context):
    del t, logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    ddx2 = directional_dx.pow(2).view(x.size(0),-1)

    return 0.5*ddx2.mean(dim=-1)

def quadratic_cost(x, t, logp, dx, dlogp, unused_context):
    del x, logp, dlogp, t, unused_context
    dx = dx.view(dx.shape[0], -1)
    return 0.5*dx.pow(2).mean(dim=-1)

def jacobian_frobenius_regularization_fn(x, t, logp, dx, dlogp, context):
    sh = x.shape
    del logp, dlogp, t, dx, x
    sqjac = context.sqjacnorm

    return context.sqjacnorm

def creat_reg_fns(args):
    REGULARIZATION_FNS = {
        "kinetic_energy": quadratic_cost,
        "jacobian_norm2": jacobian_frobenius_regularization_fn,
        "directional_penalty": directional_derivative
    }

    reg_fns = []
    reg_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            reg_fns.append(reg_fn)
            reg_coeffs.append(eval("args." + arg_key))

    reg_fns = tuple(reg_fns)
    reg_coeffs = tuple(reg_coeffs)
    return reg_fns, reg_coeffs

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn