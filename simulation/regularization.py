"""Regularizers for the conditional flow."""
def kinetic_energy(x, t, logp, dx, dlogp, context):
    return 0.5 * dx.reshape(dx.shape[0], -1).square().mean(dim=-1)


def jacobian_norm(x, t, logp, dx, dlogp, context):
    return context.sqjacnorm


def penalty_terms(config):
    return (kinetic_energy, jacobian_norm), (config.kinetic_energy, config.jacobian_norm2)
