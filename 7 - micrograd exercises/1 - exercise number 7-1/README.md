<h3>Let's take a look closer at nodes with the power rule in compute graph.</h3>
This exercise is based on an example from the PyTorch documentation, "A Gentle Introduction to torch.autograd" (https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd). Let's take a closer look at the node in the graph that performs an exponentiation operation for example, "a ** 3". 
<br /><br />
The motivation for this exercise comes from a fine explanation in the pytorch documentation. As we know from the Andrej's micrograd video, micrograd is designed to perform a correctness check that aligns with PyTorch's calculations. It demonstrated this in previous exercises. They used an example for :

```Q = 3*a**3 - b**2``` But I’ll simplify this as much as possible to start with a very simple example that’s easy to calculate by hand.
```Q = a ** 3```


<h3>Let's start</h3>

```self.grad += (other * self.data**(other-1)) * out.grad``` If we look at the definition of the `__pow__` function in the micrograd engine within the `.ipynb` file, we will find the formula for calculating the gradient for this equation. But the formula found under the term "differentiation power rule" is $\(\frac{d}{dx}(x^{n})=nx^{n-1}\)\$
<br /><br />
Okay, let's run the script from the .ipynb file to check the setup and see if it generates a test graph for d = a*b + c. It should render the graph.
<br /><br />
Now, let's set up an exercise similar to the one in the PyTorch documentation—"A Gentle Introduction to torch.autograd"—but only for Q = a ** 3.

Setup values
```
x00 = Value(2.0, label='x00')
x01 = Value(3.0, label='x01')
```

Forward and backward pass

```
x00f1 = x00 ** 3; x00f1.label = 'x00f1'
x01f1 = x01 ** 3; x01f1.label = 'x01f1'

loss = x00f1 + x01f1; loss.label = 'loss'

loss.backward()

x00 -= 0.01 * x00.grad
x01 -= 0.01 * x01.grad

draw_dot(loss)
```

We obtain a graph.

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/11dc04f5c1543365835653df9c3d1655b0fca0e9/7%20-%20micrograd%20exercises/1%20-%20exercise%20number%207-1/graph1_25-09-2026_ex.svg)

<h3>Here is the main part of this exercise.</h3>
In this exercise, I am focusing solely on the node that computes the result for __pow__. Knowing the power rule for calculating derivatives (gradients), I can manually perform the forward and backward passes for these specific nodes x00f1 and x00.
<br /><br />

1) ```formula to calc grad x00 --> n * X ** n - 1```

2) ```learnig rate lr = 0.01 in this case```

3) Okay, the first step is naive SGD, which corresponds to the parameter update for x00, meaning the expected result of this operation at node x00.data should be this value. Look at the graph below. ```2.0 - (lr * 12.0) = 1.88```

4) Ok, I have new data value. Now I can perform the forward pass for this node that is, x00.data ** 3. ```1.88 ** 3 = 6.644671999999999```

5) Now I have new x0f1 data 6.644671999999999. Next, using the power rule formula, I calculate the gradient at x00. ```3 * 1.88 ** (3 - 1) = 10.6032```

6) I get the same result by manually performing another forward pass in listing [8] in the .ipynb file. Next, it performs a forward pass with the updated parameters, calculates the new loss, and updates the parameters—including those in the node containing x00.

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/11dc04f5c1543365835653df9c3d1655b0fca0e9/7%20-%20micrograd%20exercises/1%20-%20exercise%20number%207-1/graph2_25-09-2026_ex.svg)

<h3>Pytorch version</h3>

setup ( listing [45] )

```
a = torch.tensor([2., 3.], requires_grad=True)
```

forward and backward  ( listing [47] )

```
#b = torch.tensor([6., 4.], requires_grad=True)
Q1 = a ** 3
print(a)
print(a.grad)
print(" -- Q1 --" )
print(Q1)
loss = Q1.sum()
print( " loss " , loss)
a.grad = None
loss.backward()
print(a.grad, a.data)
a.data += -0.01 * a.grad
```

output ( step 1 )

```
tensor([2., 3.], requires_grad=True)
None
 -- Q1 --
tensor([ 8., 27.], grad_fn=<PowBackward0>)
 loss  tensor(35., grad_fn=<SumBackward0>)
tensor([12., 27.]) tensor([2., 3.])
```

output ( step 2 )

```
tensor([1.8800, 2.7300], requires_grad=True)
tensor([12., 27.])
 -- Q1 --
tensor([ 6.6447, 20.3464], grad_fn=<PowBackward0>)
 loss  tensor(26.9911, grad_fn=<SumBackward0>)
tensor([10.6032, 22.3587]) tensor([1.8800, 2.7300])
```

A comparison of the same simple calculation using the PyTorch engine. After two steps, the loss is identical 26.9911. The parameters in the nodes also match.

<h3>Conclusions</h3>
This is not a sophisticated demo, but the exercise serves as an introduction to tracing and better understanding how local gradients are calculated at a node in compute graph.
<br /><br />
// 25-09-2026

<h3>References</h3>
1. https://github.com/karpathy/micrograd - micrograd repo<br />
2. https://karpathy.github.io/2026/02/12/microgpt/ - new post about micrograd <br />
3. https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd - Neural Networks: Zero to Hero - notebooks
