
### Backpropagation in PyTorch

- First initialize the tensors z,y,z as -3,5,2. However, we need to set `requires_grad` flag to True, inorder to tell PyTorch that we need their derivatives.
- Finally we write `f.backward()` to tell PyTorch to compute the derivatives.
- Results are the same as when we calculated them by hand. `tensor.grad` simply gets the gradient of that tensor.

```python
import torch

x = torch.tensor(-3., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = torch.tensor(-2., requires_grad=True)

q = x + y
f = q * z

f.backward()

print("Gradient of z is: " + str(z.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of x is: " + str(x.grad))
```

### Building a neural network - PyTorch style
- PyTorch has a better way of building neural networks, which is object-oriented. We define a class call it Net, which inherits from `nn.Module`.
- In the `__init__` method, we define our parameters, the tensors of weights. For fully connected layers, they are called `nn.Linear``. 
- The first parameter is the number of units of the current layer, while the second parameter is the number of units in the next layer.
- In the forward method, we apply all those weights to our input. Finally we instantiate our model by calling class Net, and we get the result, by applying object net over our input_layer.

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 4)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

input_layer = torch.rand(10)
net = Net()
result = net(input_layer)
```

### Neural Networks
- 

















