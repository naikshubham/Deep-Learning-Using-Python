
## Introduction to Tensorflow using Python

### Tensorflow
- Open-source library for graph based numerical computation, developed by Google Brain
- Has low and high level APIs for addition, multiplication, differentiation and building ML models.

#### Important changes in TensorFlow 2.0
- Eager execution by default
- Model building with Keras and Estimators

#### Defining tensors in TensorFlow
- **Tensor** : Generalization of vectors and matrices to potentially higher dimensions.

```python
import tensorflow as tf

# 0D tensor
d0 = tf.ones((1,))

# 1D
d1 = tf.ones((2, ))

# 2D tensor
d2 = tf.ones((2, 2))

# 3D tensor
d3 = tf.ones((2, 2, 2))

# print the 3D tensor
print(d3.numpy())
```

#### Defining constants in Tensorflow
- A constant is the simplest category of tensor. A constant does not change and cannot be trained. It can have any dimesions.
- The constant b is 2x2 tensor, which is constructed from the 1-dimensional tensor: 1,2,3,4

```python
from tensorflow import constant

# define a 2x3 constant
a = constant(3, shape=[2,3])

# Define a 2x2 constant
b = constant([1,2,3,4], shape=[2,2])
```

#### Defining and initializing variables
- Unlike a constant a variables value can be change during the computation.
- a0 is a 1-D tensor with 6 elements.

```python
import tensorflow as tf

# define a variable
a0 = tf.Variable([1,2,3,4,5,6], dtype=tf.float32)
a1 = tf.Variable([1,2,3,4,5,6], dtype=tf.int16)

# define a constant b
b = tf.constant(2, tf.float32)

# compute their product
c0 = tf.multiply(a0, b)
c1 = a0*b
```

### Basic Operations
- **Tensorflow Operation** : Tensorflow has a model of computation that revolves around the use of graphs. A Tensorflow graph contains edges and nodes, where the edges are tensors and the nodes are operations.

#### Applying the addition operator
- The `add()` operator performs element-wise addition with two tensors. Elemet-wise addition requires both tensors to have the same shape.

```python
from tensorflow import constant, add

# define 0-d tensors
A0 = constant([1])
B0 = constant([2])

A1 = constant([1,2])
B1 = constant([3,4])

A2 = constant([[1,2], [3,4]])
B2 = constant([[5,6], [7,8]])

# applying addition operator
C0 = add(A0, B0) # scalar addition
C1 = add(A1, B1) # vector addition
C2 = add(A2, B2) # matrix addition
```

#### Multiplication in TensorFlow
- Element wise multiplication performed using `multiply()` operation. The tensors multiplied must have the same shape.
- **Matrix mul** : performed with `matmul()` operator. It requires number of coulmns of A to be equal to number of rows of B.

#### Summing over tensor dimensions
- The `reduce_sum()` operator sums over the dimensions of a tensor. This can be used to sum over all dimensions of a tensor or just one.
- `reduce_sum(A)` sums over all dimensions of A, `reduce_sum(A,i)` sums over dimension i.











