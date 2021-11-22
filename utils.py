from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adagrad, Adam, RMSprop, SGD
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor


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


def load_fashionmnist(train=True, shrinkage=None):
    dataset = torchvision.datasets.FashionMNIST(
        root='.',
        download=True,
        train=train,
        transform=Compose([ToTensor(), Lambda(torch.flatten)])
    )
    if shrinkage:
        dataset_size = len(dataset)
        perm = torch.randperm(dataset_size)
        idx = perm[:int(dataset_size * shrinkage)]
        return torch.utils.data.Subset(dataset, idx)
    return dataset


class ModelTrainer:
    def __init__(self, train_dataset, test_dataset, batch_size=128):
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, model, optimizer, loss_fn=torch.nn.functional.cross_entropy, n_epochs=100):
        self.logs = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
        model = model.to(self.device)
        correct, numel = 0, 0
        for e in range(1, n_epochs + 1):
            model.train()
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                y_pred = torch.argmax(output, dim=1)
                correct += torch.sum(y_pred == y).item()
                numel += self.batch_size
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

            self.logs['train_loss'].append(loss.item())
            self.logs['train_accuracy'].append(correct / numel)
            correct, numel = 0, 0

            model.eval()
            with torch.no_grad():
                for x_test, y_test in self.test_loader:
                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)
                    output = model(x_test)
                    y_pred = torch.argmax(output, dim=1)
                    correct += torch.sum(y_pred == y_test).item()
                    numel += self.batch_size
                loss = loss_fn(output, y_test)

            self.logs['test_loss'].append(loss.item())
            self.logs['test_accuracy'].append(correct / numel)
            correct, numel = 0, 0

        return self.logs


def show_results(orientation='horizontal', accuracy_bottom=None, loss_top=None, **histories):
    if orientation == 'horizontal':
        f, ax = plt.subplots(1, 2, figsize=(16, 5))
    else:
        f, ax = plt.subplots(2, 1, figsize=(16, 16))
    for i, (name, h) in enumerate(histories.items()):
        if len(histories) == 1:
            ax[0].set_title("Best test accuracy: {:.2f}% (train: {:.2f}%)".format(
                max(h['test_accuracy']) * 100,
                max(h['train_accuracy']) * 100
            ))
        else:
            ax[0].set_title("Accuracy")
        ax[0].plot(h['train_accuracy'], color='C%s' % i, linestyle='--', label='%s train' % name)
        ax[0].plot(h['test_accuracy'], color='C%s' % i, label='%s test' % name)
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('accuracy')
        if accuracy_bottom:
            ax[0].set_ylim(bottom=accuracy_bottom)
        ax[0].legend()

        if len(histories) == 1:
            ax[1].set_title("Minimal train loss: {:.4f} (test: {:.4f})".format(
                min(h['train_loss']),
                min(h['test_loss'])
            ))
        else:
            ax[1].set_title("Loss")
        ax[1].plot(h['train_loss'], color='C%s' % i, linestyle='--', label='%s train' % name)
        ax[1].plot(h['test_loss'], color='C%s' % i, label='%s test' % name)
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('loss')
        if loss_top:
            ax[1].set_ylim(top=loss_top)
        ax[1].legend()

    plt.show()


def test_dropout(dropout_cls):
    drop = dropout_cls(0.5)
    drop.train()
    x = torch.randn(10, 30)
    out = drop(x)

    for row in out:
        zeros_in_row = len(torch.where(row == 0.)[0])
        assert 0 < zeros_in_row < len(row)

    drop_eval = dropout_cls(0.5)
    drop_eval.eval()
    x = torch.randn(10, 30)
    out_eval = drop_eval(x)

    for row in out_eval:
        zeros_in_row = len(torch.where(row == 0.)[0])
        assert zeros_in_row == 0


def test_bn(bn_cls):
    torch.manual_seed(42)
    bn = bn_cls(num_features=100)

    opt = torch.optim.SGD(bn.parameters(), lr=0.1)

    bn.train()
    x = torch.rand(20, 100)
    out = bn(x)

    assert out.mean().abs().item() < 1e-4
    assert abs(out.var().item() - 1) < 1e-1

    assert (bn.sigma != 1).all()
    assert (bn.mu != 1).all()

    loss = 1 - out.mean()
    loss.backward()
    opt.step()

    assert (bn.beta != 0).all()

    n_steps = 10

    for i in range(n_steps):
        x = torch.rand(20, 100)
        out = bn(x)
        loss = 1 - out.mean()
        loss.backward()
        opt.step()

    torch.manual_seed(43)
    test_x = torch.randn(20, 100)
    bn.eval()
    test_out = bn(test_x)

    assert abs(test_out.mean() + 0.5) < 1e-1
