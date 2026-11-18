<h3>Description for exercise 6-1</h3>

The main goal is to understand what lies behind optimizers.
1. https://www.tensorflow.org/guide/core/optimizers_core
2. https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
<br /><br />
3. https://web.stanford.edu/~jurafsky/slp3/old_dec21/7.pdf - (Neural Networks and NeuralLanguage Models) - For informational purposes only, to provide a broader perspective.
<br /><br />
<h4>About the notebook file</h4>

File `Untitled1 - 18-11-2026 - ex_6-1.ipynb` .
<br /><br />
What exactly is a linear operation in neural networks? What precisely is `X W + B`? etc.
<br /><br />
It is best to start with a simple example that corresponds to `xw+b`, the linear function `y = 3 * x + 2`. This is a straightforward when we know `y`. In the context of a neural network (NN), this is the `target`, `label`, or `class`. So if `y = 7`, then `7 = 3 * x + 2`, what is `x`?
<br /><br />
And this is precisely where the file `Untitled1 - 18-11-2026 - ex_6-1.ipynb` begins. In the initial code blocks, I attempt to determine this value manually by trial and error and subsequently calculate it using `loss_fn` functions:
1.  y - x
2. (y - x) ** 2 ` # MSE, from code block [32] `
3. Cross Entropy, ` from code block [44] `

<h4>Summary </h4>

Okay, this is naive and weak, but exercise ` 6-1 ` is just there to provide a starting point. 
