
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

#### Advanced operations

<img src="data/advanced_operations.JPG" width="350" title="advanced_operations">

#### Finding the optimum
- In many ML problems, we need to find an optimum, i.e, a minimum or maximum.We may,want to find the model parameters that minimize the loss function or maximize the objective function.
- Fortunately, we need to do this by using the gradient operation, which tells us the slope of a function at a point. We start this process by passing points to the gradient operation unitl we find one where gradient is zero.
- Next we check if the gradient is increasing or decreasing at that point.If its increasing, we have minimum (change in gradient >0), otherwise we have maximum (change in gradient < 0)

#### Gradients in TensorFlow
- We will start by defining a variable x, which we initialize to -1.0. We then define y=x^2 within the instance of gradient tape.
- Apply watch method to an instance of gradient tape and then pass the variable x. This will allow us to compute the rate of change of y w.r.t x.
- Next we compute the gradient of y w.r.t x using the tape instance of gradient tape. The operation computes the slope of y at a point.
- Much of the differentiation we do in DL models will be handled by high level APIs, however gradient tape remains an invaluable tool for building advanced and custom models.

```python
import tensorflow as tf

# define x
x = tf.Variable(-1.0)

# define y within instance of GradientTape
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x,x)
    
# evaluate the gradient of y at x=-1
g = tape.gradient(y, x)
print(g.numpy())
```

#### Reshape (Image as tensors)
- Some algorithms require us to **reshape matrices into vectors before using them as inputs**.
- Create random grayscale image. Use it to populate a 2x2 matrix. We can then reshape it into a (4,1) vector.
- For color images generate 3 such matrices to form a (2x2x3) tensor, then reshape it into (4,3) tensor.

```python
import tensorflow as tf

# generate grayscale image
gray = tf.random.uniform([2,2], maxval=255, dtype='int32')

# reshape grayscale image
gray = tf.reshape(gray, [2*2, 1])

# generate color image
color = tf.random.uniform([2,2,3], maxval=255, dtype='int32')

# reshape color image
color = tf.reshape(color, [2*2, 3])
```

## Linear Models with TF

### Input data
- Import data using pandas, convert data to numpy array, which can be used without modification in TensorFlow.

```python
import numpy as np
import pandas as pd

# load data from csv
housing = pd.read_csv('kc_housing.csv')

# convert to numpy array
housing = np.array(housing)
```

#### Using mixed type datasets
- Datasets with different datatype columns datetime, float, boolean. `tf.cast or np.array` can be used to convert the datatypes.

```python
price = tf.cast(housing['price'], tf.float32)
waterfront = tf.cast(housing['waterfront'], tf.bool)

# or
price = np.array(housing['price'], np.float32)
waterfront = np.array(housing['waterfront'], np.bool)
```

### Loss functions
- Tensorflow has operations for common loss functions. Typical choices for training linear models include the **Mean squared error loss(MSE), the Mean absolute error loss(MAE) and the Huber loss**
- All these loss functions are accessible from `tf.keras.losses()`, **`tf.keras.losses.mse(), tf.keras.losses.mae(), tf.keras.losses.Huber()`**

#### Why do we care about loss functions?
- **MSE** : strongly penalizes outliers and has high(gradient) sensitivity near the minimum.
- **MAE** : Scales linearly with the size of error and has low sensitivity near minimum.
- For greater sensitivity near the minimum, we will use MAE or Huber loss. To minimize the impact of outliers, we will use the MSE or HUber loss.

#### Defining a loss function
- To use the loss we will need two tensors to compute it: the actual values or 'targets' tensor and the predicted values or 'predictions'. Passing them to the MSR operation will return a single number: the average of the squared differences between the actual and predicted values.

```python
import tensorflow as tf

# compute the MSE loss
loss = tf.keras.losses.mse(targets, predictions)
```

#### Linear regression model
- Define linear regression function that accepts intercept, slope and features and returns a prediction.
- Next define a loss function that accepts the slope and intercept of a linear model , the variables, and the input data, the targets and the features. It then makes a prediction and computes and returns the associated MSE loss.

```python
# define a linear regression model

def linear_regression(intercept, slope=slope, features=features):
    return intercept + features * slope
    
# define a loss function to compute the MSE
def loss_function(intercept, slope, targets = targets, features = features):
    # compute the predictions for a linear model
    predictions = linear_regression(intercept, slope)
    
    # return the loss
    return tf.keras.losses.mse(targets, predictions)
    
# evaluate the loss function using a test dataset
loss_function(intercept, slope, test_targets, test_features)
```

### Training a Linear Regression model in TensorFlow
- A linear regression model assumes a linear relationship. `price = intercept + size * slope + error`. The difference between the predicted and the actual price is the error, which can be used to construct a loss function.
- Univariate regression : there is only one feature. Mutiple regression models have more than one feature.
- Initialize target and features. **Also initialize the intercept and slope as trainable variables.**
- After that we **define a model which will be used to make prediction by multiplying size and slope and then adding the intercept**.
- Next step is to **define a loss function**. This function will take the model's parameters and the data as an input. We first use the model to predict the values and then set the function to return the mean squared error loss.
- With loss function defined, the next step is to **define an optimization operation,**  using adam optimizer. Executing this operation will change the slope and intercept in a direction that will lower the value of the loss.
- We will next perform **minimization on the loss function using the optimizer**. Notice that we've passed the loss function as a lambda function to the minimize operation. We also supplied a variable list, which contains intercept and slope.
- The optimization step is executed 1000 times.

```python
# define the targets and features
price = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)

# define the intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

# define a linear regression model
def linear_regression(intercept, slope, features=size):
    return intercept + features * slope    
    
# compute the predicted values and loss
def loss_function(intercept, slope, targets=price, features=size):
    predictions = linear_regression(intercept, slope)
    return tf.keras.losses.mse(targets, predictions)
    
# define an optimization operation
opt = tf.keras.optimizers.Adam()

# minimize the loss functions and print the loss
for j in range(1000):
    opt.minimize(lambda: loss_function(intercept, slope), var_list=['intercept,slope'])
    print(loss_function(intercept, slope))
    
# print the trained parameters
print(intercept.numpy(), slope.numpy())
```

### Batch training
- Batch training to handle large datasets. If the dataset is very large and we want to perform training on the GPU which has only small amount of memory. So we can't fit the entire dataset in memory, we will instead divide it into batches and train on those batches sequentially.
- A single pass over all of the batches is called an epoch and the process itself is called batch training.
- Beyond alleviating memory constraints, batch training will also allow us to update model weights and optimizer parameters after each batch, rather than at the end of the epoch.

#### The chunksize parameter
- `pd.read_csv()` allows us the load data in batches using chunksize parameter

```python
import pandas as pd
import numpy as np

for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    price = np.array(batch['price'], np.float32)
    # extract size col
    size = np.array(batch['size'], np.float32)
```

#### Training a linear model in batches
- Define variables for intercept and slope along with the regression models. After defining the loss function, we instantiate an adam optimizer, which we use to perform minimization.
- Next step is to train the model in batches.
- Within the minimize operation, we pass the loss function as a lambda function and **we supply a variable list that contains only the trainable parameters, intercept and slope.**

```python
import tensorflow as tf
import pandas as pd
import numpy as np

# define trainable variables
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

# define the model
def linear_regression(intercept, slope, features):
    return intercept + features * slope
    
# compute predicted values and return loss function
def loss_function(intercept, slope, targets, features):
    predictions = linear_regression(intercept, slope, target)
    return tf.keras.losses.mse(targets, predictions)
    
# define optimization operation
opt = tf.keras.optimizers.Adam()

# load the data in batches from pandas
for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    # extract the target & features columns
    price_batch = np.array(batch['price'], np.float32)
    size_batch = np.array(batch['lot_size'], np.float32)
    
    # minimize the loss function
    opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])
    
# print trained interecept and slope
print(intercept.numpy() , slope.numpy())
```

### Dense layers
- How do we get from a linear regression to a neural network? By adding a hidden layer, which, in this case, consists of two nodes. Each hidden layer node takes the two inputs, multiplies them by their respective weights, and sums them together.
- Here we will learn to construct neural networks with 3 types of layers: an input layer, some number of hidden layers, and an output layer. 
- The input layer consists of our features. The output layer contains our prediction. Each hidden layer takes inputs from the previous layer, applies numerical weights to them, sums them together, and then applies an activation function.
- In the neural network graph, we have applied a particular type of hidden layer called a dense layer. A dense layer is a fully connected layer and applies weights to all nodes from the previous layer.

#### A simple dense layer
- First define a constant tensor that contains the marital status and age data as the input layer.
- We then initialize weights as a variable, since we will train those weights to predict the output from the inputs.
- We also define a bias, which will pay a similar role to the intercept in the linear regression model.
- Finally we define a dense layer. We first perform a matrix mul of the inputs by the weights and assign that to the tensor named product. We then add product to the bias and apply a non-linear transformation, in this case the sigmoid function. This is called the activation function.
- The bias is not associated with the feature and is analogous to the intercept in a linear regression.

```python
import tensorflow as tf

# define inputs (features)
inputs = tf.constant([[1, 35]])

# define weights
weights = tf.Variable([[-0.05], [-0.01]])

# define the bias
bias = tf.Variable([0.5])

# multiply inputs (features) by the weights
product = tf.matmul(inputs, weights)

# define dense layer
dense = tf.keras.activations.sigmoid(product + bias)
```

#### Defining a complete model
- Tensorflow also comes with higher level operations, such as `tf.keras.layers.Dense`, which allows us to skip the linear algebra.
- In the dense layer, the first argument specifies the number of outgoing nodes. By default, a bias will be included. We've also passed inputs as an argument to the first dense layer.
- We can easily define another dense layer, which takes the first dense layer as an argument and then reduces the number of nodes.
- The output layer reduces this again to one node.

```python
import tensorflow as tf

# define input (features) layer
inputs = tf.constant(data, tf.float32)

# define first dense layer
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

# define second dense layer
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)

# define output (predictions) layer
outputs = tf.keras.layers.Dense(1, activation='sigmod')(dense2)
```




























