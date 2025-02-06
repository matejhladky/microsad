import numpy as np
from tensor import Tensor

def test_scalar_add():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.bprop()

    assert np.allclose(a.grad, 1.0), f"Expected 1.0, got {a.grad}"
    assert np.allclose(b.grad, 1.0), f"Expected 1.0, got {b.grad}"
    print("Scalar addition OK")

def test_vector_add():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b
    c.bprop()

    expected = np.array([1.0, 1.0])
    assert np.allclose(a.grad, expected), f"Expected {expected}, got {a.grad}"
    assert np.allclose(b.grad, expected), f"Expected {expected}, got {b.grad}"
    print("Vector addition OK")

def test_grad_accumulation():
    a = Tensor(2.0)
    b = a + a
    b.bprop()

    assert np.allclose(a.grad, 2.0), f"Accumulation failed: {a.grad}"
    print("Gradient accumulation OK")

if __name__ == "__main__":
    test_scalar_add()
    test_vector_add()
    test_grad_accumulation()