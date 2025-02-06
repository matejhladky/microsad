from tensor import Tensor

class Op:
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
    

class Add(Op):
    def forward(self, a, b):
        self.a = a
        self.b = b
        return Tensor(a.data + b.data, op=self)