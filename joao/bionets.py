import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

class StatelessNetError(Exception):
    pass

class Net(ABC):
    def __init__(self, config):
        self.parameters = {}
        self.parameters['W1'] = config['w-init'] * torch.randn(config['n1'], 28*28, device=config['device'])
        self.parameters['b1'] = config['w-init'] * torch.randn(config['n1'], device=config['device'])
        self.parameters['W2'] = config['w-init'] * torch.randn(config['n2'], config['n1'], device=config['device'])
        self.parameters['b2'] = config['w-init'] * torch.randn(config['n2'], device=config['device'])
        self.parameters['W3'] = config['w-init'] * torch.randn(10, config['n2'], device=config['device'])
        self.parameters['b3'] = config['w-init'] * torch.randn(10, device=config['device'])

        self.parameters['B2'] = config['b-init'] * (torch.rand(config['n2'], 10, device=config['device']) - 0.5)
        self.parameters['B1'] = config['b-init'] * (torch.rand(config['n1'], config['n2'], device=config['device']) - 0.5)

        self.vars = {}
        self.reset_state()

    def reset_state(self):
        self.vars['r0'] = None
        self.vars['r1'] = None
        self.vars['r2'] = None
        self.vars['r3'] = None

    def forward(self, x):
        self.vars['r0'] = x.view(-1, 28*28)
        self.vars['r1'] = torch.sigmoid(self.vars['r0'].mm(self.parameters['W1'].t()) + self.parameters['b1'])
        self.vars['r2'] = torch.sigmoid(self.vars['r1'].mm(self.parameters['W2'].t()) + self.parameters['b2'])
        self.vars['r3'] = torch.sigmoid(self.vars['r2'].mm(self.parameters['W3'].t()) + self.parameters['b3'])

        return self.vars['r3']

    def loss_function(self, target):
        loss_function = nn.CrossEntropyLoss()
        output = self.vars['r3']
        return loss_function(output, target).item()

    def learn(self, config, target_categorical):
        if self.vars['r0'] is None:
            raise StatelessNetError('Uninitialized network.')

    def get_numpy_parameters(self):
        parameters = {}
        for param_name, param in self.parameters.items():
            parameters[param_name] = self.parameters[param_name].cpu().numpy().copy()
        return parameters

class FANet(Net):
    def learn(self, config, target_categorical):
        super().learn(config, target_categorical)

        # Convert target_categorical to a one-hot encoded pattern of activity.
        target = torch.empty(target_categorical.shape[0], 10, device=config['device'])
        target.zero_()
        target.scatter_(1, target_categorical.unsqueeze(1), 1)

        # Derivative of neuron-by-neuron cross-entropy loss wrt to output activations.
        e3 = target - self.vars['r3']
        # Efficiently compute average of outer products.
        dW3 = torch.sum(torch.bmm(e3.unsqueeze(2), self.vars['r2'].unsqueeze(1)), dim=0)
        db3 = torch.sum(e3, dim=0)

        dr2 = self.vars['r2'] * (1 - self.vars['r2'])
        # 'Backpropagate' errors.
        e2 = dr2 * e3.mm(self.parameters['B2'].t())
        dW2 = torch.sum(torch.bmm(e2.unsqueeze(2), self.vars['r1'].unsqueeze(1)), dim=0)
        db2 = torch.sum(e2, dim=0)

        dr1 = self.vars['r1'] * (1 - self.vars['r1'])
        e1 = dr1 * e2.mm(self.parameters['B1'].t())
        dW1 = torch.sum(torch.bmm(e1.unsqueeze(2), self.vars['r0'].unsqueeze(1)), dim=0)
        db1 = torch.sum(e1, dim=0)

        # In FA only forward weights are learned.
        self.parameters['W3'] += config['eta3']/config['batch-size'] * dW3
        self.parameters['b3'] += config['eta3']/config['batch-size'] * db3
        self.parameters['W2'] += config['eta2']/config['batch-size'] * dW2
        self.parameters['b2'] += config['eta2']/config['batch-size'] * db2
        self.parameters['W1'] += config['eta1']/config['batch-size'] * dW1
        self.parameters['b1'] += config['eta1']/config['batch-size'] * db1
        
class TPNet(Net):
    def learn(self, config, target_categorical):
        super().learn(config, target_categorical)

        # Convert target_categorical to a one-hot encoded pattern of activity.
        target = torch.empty(target_categorical.shape[0], 10, device=config['device'])
        target.zero_()
        target.scatter_(1, target_categorical.unsqueeze(1), 1)

        target = (1 - config['lambda'])*self.vars['r3'] + config['lambda']*target
        # Derivative of neuron-by-neuron cross-entropy loss wrt to output activations.
        e3 = target - self.vars['r3']
        # Efficiently compute average of outer products.
        dW3 = torch.sum(torch.bmm(e3.unsqueeze(2), self.vars['r2'].unsqueeze(1)), dim=0)
        db3 = torch.sum(e3, dim=0)

        dr2 = self.vars['r2'] * (1 - self.vars['r2'])
        if config['diff-tp']:
            e2 = dr2 * e3.mm(self.parameters['B2'].t())
        else:
            target2 = (1 - config['lambda'])*self.vars['r2'] + config['lambda']*target.mm(self.parameters['B2'].t())
            e2 = dr2 * (target2 - self.vars['r2'])
            
        dW2 = torch.sum(torch.bmm(e2.unsqueeze(2), self.vars['r1'].unsqueeze(1)), dim=0)
        db2 = torch.sum(e2, dim=0)

        dr1 = self.vars['r1'] * (1 - self.vars['r1'])
        if config['diff-tp']:
            e1 = dr1 * e2.mm(self.parameters['B1'].t())
        else:
            target1 = (1 - config['lambda'])*self.vars['r1'] + config['lambda']*target2.mm(self.parameters['B1'].t())
            e1 = dr1 * (target1 - self.vars['r1'])
        dW1 = torch.sum(torch.bmm(e1.unsqueeze(2), self.vars['r0'].unsqueeze(1)), dim=0)
        db1 = torch.sum(e1, dim=0)

        # Learn backward weights with a reverse delta rule (without using information from the labels)
        r2_b = torch.sigmoid(self.vars['r3'].mm(self.parameters['B2'].t()))
        e2_b = self.vars['r2'] - r2_b
        dB2 = torch.sum(torch.bmm(e2_b.unsqueeze(2), self.vars['r3'].unsqueeze(1)), dim=0)

        r1_b = torch.sigmoid(self.vars['r2'].mm(self.parameters['B1'].t()))
        e1_b = self.vars['r1'] - r1_b
        dB1 = torch.sum(torch.bmm(e1_b.unsqueeze(2), self.vars['r2'].unsqueeze(1)), dim=0)

        self.parameters['W3'] += config['eta3']/config['batch-size'] * dW3
        self.parameters['b3'] += config['eta3']/config['batch-size'] * db3
        self.parameters['W2'] += config['eta2']/config['batch-size'] * dW2
        self.parameters['b2'] += config['eta2']/config['batch-size'] * db2
        self.parameters['W1'] += config['eta1']/config['batch-size'] * dW1
        self.parameters['b1'] += config['eta1']/config['batch-size'] * db1
        self.parameters['B2'] += config['eta-b2']/config['batch-size'] * dB2
        self.parameters['B1'] += config['eta-b1']/config['batch-size'] * dB1

class Net1(ABC):
    def __init__(self, config):
        self.parameters = {}
        self.parameters['W1'] = config['w-init'] * torch.randn(config['n1'], 28*28, device=config['device'])
        self.parameters['b1'] = config['w-init'] * torch.randn(config['n1'], device=config['device'])
        self.parameters['W2'] = config['w-init'] * torch.randn(config['n2'], config['n1'], device=config['device'])
        self.parameters['b2'] = config['w-init'] * torch.randn(config['n2'], device=config['device'])


        self.parameters['B1'] = config['b-init'] * (torch.rand(config['n1'], config['n2'], device=config['device']) - 0.5)

        self.vars = {}
        self.reset_state()
        self.slope = 0.01
        self.config = config

    def reset_state(self):
        self.vars['r0'] = None
        self.vars['r1'] = None
        self.vars['r2'] = None

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(self.slope)
        self.vars['r0'] = x.view(-1, 28*28)
        self.vars['r1'] = leaky_relu(self.vars['r0'].mm(self.parameters['W1'].t()) + self.parameters['b1'])
        self.vars['r2'] = leaky_relu(self.vars['r1'].mm(self.parameters['W2'].t()) + self.parameters['b2'])
        return self.vars['r2']

    def learn(self, config, target_categorical):
        if self.vars['r0'] is None:
            raise StatelessNetError('Uninitialized network.')

    def get_numpy_parameters(self):
        parameters = {}
        for param_name, param in self.parameters.items():
            parameters[param_name] = self.parameters[param_name].cpu().numpy().copy()
        return parameters

    def jac(self,activation):
        jac = torch.empty(activation.shape,
                             device=self.config['device'])
        for b in range(activation.shape[0]):
            for i in range(activation.shape[1]):
                if activation[b,i] < 0:
                    jac[b,i] = self.slope
                else:
                    jac[b,i] = 1.

        return jac

class CapsuleNetBP(Net1):
    def learn(self, config, target_categorical):
        super().learn(config, target_categorical)

        # Convert target_categorical to a one-hot encoded pattern of activity.
        target = torch.empty(target_categorical.shape[0], 10,
                             device=config['device'])
        target.zero_()
        target.scatter_(1, target_categorical.unsqueeze(1), 1)

        # Derivative of neuron-by-neuron cross-entropy loss wrt to output activations.
        e2 = self.compute_capsule_error(target)

        # Efficiently compute average of outer products.
        dW2 = torch.sum(
            torch.bmm(e2.unsqueeze(2), self.vars['r1'].unsqueeze(1)), dim=0)
        db2 = torch.sum(e2, dim=0)

        dr1 = self.jac(self.vars['r1'])
        e1 = dr1 * e2.mm(self.parameters['W2'])
        dW1 = torch.sum(
            torch.bmm(e1.unsqueeze(2), self.vars['r0'].unsqueeze(1)), dim=0)
        db1 = torch.sum(e1, dim=0)

        # In BP only forward weights are learned.
        self.parameters['W2'] += config['eta2'] / config['batch-size'] * dW2
        self.parameters['b2'] += config['eta2'] / config['batch-size'] * db2
        self.parameters['W1'] += config['eta1'] / config['batch-size'] * dW1
        self.parameters['b1'] += config['eta1'] / config['batch-size'] * db1

    def compute_capsules(self):
        config = self.config
        excess = config['n2'] % 10
        capsule_base_size = int(config['n2'] / 10)
        capsule_indices = {}
        capsule_magnitudes = torch.zeros(self.vars['r0'].shape[0], 10,
                             device=self.config['device'])
        start = 0
        for capsule in range(10):
            if capsule < excess:
                stop = start + capsule_base_size + 1
            else:
                stop = start + capsule_base_size
            capsule_indices[capsule] = (start, stop)

            # compute magnitude capsules
            capsule_magnitudes[:, capsule] = torch.norm(
                self.vars['r2'][:, start:stop], dim=1
            )
            # initialize start for next iteration
            start = stop

        capsule_squashed = capsule_magnitudes ** 2 / (
                    1 + capsule_magnitudes ** 2)

        self.capsule_indices = capsule_indices
        self.vars['capsule_magnitudes'] = capsule_magnitudes
        self.vars['capsule_squashed'] = capsule_squashed

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(self.slope)
        self.vars['r0'] = x.view(-1, 28*28)
        self.vars['r1'] = leaky_relu(self.vars['r0'].mm(self.parameters['W1'].t()) + self.parameters['b1'])
        self.vars['r2'] = leaky_relu(self.vars['r1'].mm(self.parameters['W2'].t()) + self.parameters['b2'])
        self.compute_capsules()
        return self.vars['capsule_squashed']

    def compute_capsule_error(self, target):
        l = 0.5
        m_plus = 0.9
        m_min = 0.1
        capsule_indices = self.capsule_indices
        capsule_squashed = self.vars['capsule_squashed']
        capsule_magnitudes = self.vars['capsule_magnitudes']

        # compute loss gradient
        gradient = torch.empty(self.vars['r2'].shape,
                             device=self.config['device'])
        Lk_vk = -2 * target * \
                torch.max(torch.stack([m_plus - capsule_squashed,
                                       torch.zeros(
                                           capsule_squashed.shape,
                             device=self.config['device'])]),
                          dim=0)[0] + \
                2 * l * (1 - target) * \
                torch.max(torch.stack([capsule_squashed \
                                       - m_min,
                                       torch.zeros(
                                           capsule_squashed.shape,
                             device=self.config['device'])]),
                          dim=0)[0]

        for capsule in range(10):
            start = capsule_indices[capsule][0]
            stop = capsule_indices[capsule][1]
            vk_sk = 1/((1+capsule_magnitudes[:,capsule].unsqueeze(1)**2)**2)\
                    * 2*self.vars['r2'][:, start:stop]

            gradient[:, start:stop] = Lk_vk[:, capsule].unsqueeze(1) * vk_sk

        return -gradient

    def loss_function(self, target_categorical):
        target = torch.empty(target_categorical.shape[0], 10,
                             device=self.config['device'])
        target.zero_()
        target.scatter_(1, target_categorical.unsqueeze(1), 1)
        l = 0.5
        m_plus = 0.9
        m_min = 0.1
        capsule_squashed = self.vars['capsule_squashed']

        # compute hinton loss
        L_k = target * \
              torch.max(torch.stack([m_plus - capsule_squashed,
                                     torch.zeros(
                                         capsule_squashed.shape,
                             device=self.config['device'])]),
                        dim=0)[0] ** 2 + \
              l * (1 - target) * \
              torch.max(torch.stack([capsule_squashed - m_min,
                                     torch.zeros(
                                         capsule_squashed.shape,
                             device=self.config['device'])]),
                        dim=0)[0] ** 2
        loss = torch.sum(L_k)
        return loss







