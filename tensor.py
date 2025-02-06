import numpy as np
from op import Add

class Tensor(np.ndarray):
    def __new__(cls, data, op=None):
        obj = np.asarray(data).view(cls)
        obj.op = op
        obj.grad = np.zeros(obj.shape, dtype=obj.dtype)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.op = getattr(obj, 'op', None)
        self.grad = getattr(obj, 'grad', np.zeros(self.shape, dtype=self.dtype))

    def bprop(self):
        visited = set()
        topo = []

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                if node.op:
                    for inp in node.op.inputs:
                        build_topo(inp)
                topo.append(node)
        
        build_topo(self)
        
        self.grad = np.ones(self.shape, dtype=self.dtype)
        
        for node in reversed(topo):
            if node.op:
                node.op.backward(node.grad)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add().forward(self, other)
    
    
    def __repr__(self):
        return f"<Tensor {np.ndarray.__repr__(self)}>"