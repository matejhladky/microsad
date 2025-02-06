import tensor
import numpy as np

class Op:
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Add(Op):
    def forward(self, a, b):
        self.inputs = (a, b)
        return tensor.Tensor(np.asarray(a) + np.asarray(b), op=self)
    
    def backward(self, grad):
        a, b = self.inputs
        a.grad += grad
        b.grad += grad
