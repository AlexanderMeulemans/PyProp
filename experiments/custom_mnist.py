"""
Copyright 2019 Alexander Meulemans

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
"""

import torch

# User variables
l1 = 28*28
l2 = 28*28
l3 = 10

W1 = torch.randn(l2,l1)
W2 = torch.randn(l3,l2)

b1 = torch.randn(l2,1)
b2 = torch.randn(l3,1)

def forward(x):
    a1 = torch.matmul(W1,x) + b1
    h1 = torch.max(torch.stack([a1, torch.zeros(a1.size)]))[0]
    a2 = torch.matmul(W2, h1) + b2