
<h3>Fist exercise - fake iris dataset for manual implementation vs pytorch to check correctness of network implementation and calculations</h3>
Andrej did the same thing in his micrograd video on YouTube. This is specifically for this repo: https://github.com/karpathy/micrograd . But I started not with "make moons" but with something else that seems easier to analyze for me. But that's in the next demo - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
<br /><br />
Andrej in the first micrograd recording https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ He showed a lot more, primarily by hand-writing  autograd engine. And a few other details. But the whole thing tried to imitate the Pytorch API. And this "from scratch" network implementation is also compared to Pytorch, with some functions, as in my case https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss .There's something else going on, but the key to this exercise is, among other things, creating your own implementation that achieves the same results as a working pytorch. And that's the first exercise here.
<br /><br />
<b>See the file exercise_1_29_04_2026_micrograd_.ipynb</b>
<br /><br />
Loss value for manual calculation ( first lising in exercise_1_29_04_2026_micrograd_.ipynb file  )- final loss  tensor(0.1116)
<br /><br />
Correctness test using pytorch API where is loss.backward() - tensor(0.1116, grad_fn=<BinaryCrossEntropyBackward0>)
<br /><br />
In both cases the losses are the same. There are some differences in prediction because pytorch has grad.zero_ and the manual implementation does not reset the values ​​for dw, db, dz, therefore the value of "p" in prediction may be different than at the beginning of the training loop, which is not the case in the pytorch implementation. But this is a small detail.
<br /><br />
Tiny fake flower dataset - these are simply 4 examples with 2 features each, and described classes 0 and 1. That's all for now. <b>The goal was to make something small enough that it could be easily developed and easy to understand basic calculations.</b>
<br /><br />
// 29.04.2026
