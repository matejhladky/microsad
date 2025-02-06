import numpy as np
from op import Add

class Tensor:
    def __init__(self, data, op=''):
        self.data = np.array(data)
        self.op = op

    def get_op(self):
        return self.op

    def __add__(self, other):
        return Add().forward(self, other)
    
    def __repr__(self):
        return f'<tensor({self.data})>'