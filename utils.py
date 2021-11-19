from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adagrad, Adam, RMSprop, SGD


def visualize_optimizer(optim, n_steps, title=None, **params):
    def f(w):
        x = torch.tensor([0.2, 2], dtype=torch.float)
        return torch.sum(x * w ** 2)

    w = torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)

    optimizer = optim([w], **params)

    history = [w.clone().detach().numpy()]

    for i in range(n_steps):
        optimizer.zero_grad()

        loss = f(w)
        loss.backward()
        optimizer.step()
        history.append(w.clone().detach().numpy())

    delta = 0.01
    x = np.arange(-7.0, 7.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)

    Z = 0.2 * X ** 2 + 2 * Y ** 2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.contour(X, Y, Z, 20)

    h = np.array(history)

    ax.plot(h[:, 0], h[:, 1], 'x-')

    if title is not None:
        ax.set_title(title)


def optim_f(w):
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w ** 2)


def optim_g(w, b):
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w + b)


opt_checker_1 = SimpleNamespace(f=optim_f,
                                params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)])

opt_checker_2 = SimpleNamespace(f=optim_g,
                                params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True),
                                        torch.tensor([1, -1], dtype=torch.float, requires_grad=True)])

test_params = {
    'GradientDescent': {
        'torch_cls': SGD,
        'torch_params': {'lr': 0.1},
        'params': {'learning_rate': 0.1}},
    'Momentum': {
        'torch_cls': SGD,
        'torch_params': {'lr': 0.1, 'momentum': 0.9},
        'params': {'learning_rate': 0.1, 'gamma': 0.9}},
    'Adagrad': {'torch_cls': Adagrad,
                'torch_params': {'lr': 0.5, 'eps': 1e-8},
                'params': {'learning_rate': 0.5, 'epsilon': 1e-8}},
    'RMSProp': {'torch_cls': RMSprop,
                'torch_params': {'lr': 0.5, 'alpha': 0.9, 'eps': 1e-08, },
                'params': {'learning_rate': 0.5, 'gamma': 0.9, 'epsilon': 1e-8}},
    'Adam': {'torch_cls': Adam,
             'torch_params': {'lr': 0.5, 'betas': (0.9, 0.999), 'eps': 1e-08},
             'params': {'learning_rate': 0.5, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}}}


def test_optimizer(optim_cls, num_steps=10):
    test_dict = test_params[optim_cls.__name__]

    for ns in [opt_checker_1, opt_checker_2]:

        torch_params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        torch_opt = test_dict['torch_cls'](torch_params, **test_dict['torch_params'])
        for _ in range(num_steps):
            torch_opt.zero_grad()
            loss = ns.f(*torch_params)
            loss.backward()
            torch_opt.step()

        params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        opt = optim_cls(params, **test_dict['params'])
        for _ in range(num_steps):
            opt.zero_grad()
            loss = ns.f(*params)
            loss.backward()
            opt.step()

        for p, tp in zip(params, torch_params):
            assert torch.allclose(p, tp)
