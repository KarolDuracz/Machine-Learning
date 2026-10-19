<h3>About F.cross_entropy (PyTorch) vs micrograd</h3>

The goal of this exercise is to create the simplest possible example that helps in understanding how `F.cross_entropy` works. This is equivalent to `CrossEntropyLoss()`.
<br />
1. https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html <br />
2. https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
<br /><br />
Here is an example of implementing cross-entropy from scratch https://gist.github.com/yang-zhang/217dcc6ae9171d7a46ce42e215c1fee0 . We obtain the same result as with `F.cross_entropy` or `CrossEntropyLoss()` .
<h4>Step 1</h4>
In `Untitled - 19-10-2026 - about F cross entropy loss function.ipynb` file, the first few code listings are, in fact, the execution of this comparison, taken from the Yang-Zhang notebook. The most important part of this is at the end: [18]`loss == F.cross_entropy(x, target)`
evaluates to `True`
, which means everything is correct the results from the raw implementation and `F.cross_entropy` are identical.
<h4>Step 2</h4>
Okay, let’s move on to listing [22], where I’ve added a few `print` statements to these functions so we can track what’s happening with these numbers, step by step. Next, I create a simple demo that constructs `xs`, consisting of two inputs, each containing a vector of 10 parameters. For each vector, two classes are defined as `ys`. I chose `torch.manual_seed(3409)` simply because the numbers look nice. <br /> <br />


```
torch.manual_seed(3409)

xs = torch.randn(2,10, requires_grad=True)
ys = torch.tensor([9,4])
pred = softmax(xs)
loss=nl(pred, ys)
loss.backward()
```

Output

```
 raw  tensor([[ 0.2477,  2.1216, -0.2116,  0.4789, -0.4290,  1.0829,  1.6881, -0.7816,
         -0.0902,  0.2235],
        [-0.5434, -0.8391, -1.9117, -1.9223, -0.4209, -0.2874,  0.7715,  0.7912,
          0.2874, -0.4553]], requires_grad=True)
 exp  tensor([[1.2811, 8.3445, 0.8093, 1.6143, 0.6511, 2.9532, 5.4089, 0.4577, 0.9137,
         1.2504],
        [0.5808, 0.4321, 0.1478, 0.1463, 0.6565, 0.7502, 2.1630, 2.2060, 1.3330,
         0.6342]], grad_fn=<ExpBackward0>)
 sum  tensor(32.7341, grad_fn=<SumBackward0>)
 sum(-1)  tensor([23.6842,  9.0499], grad_fn=<SumBackward1>)
tensor([[23.6842],
        [ 9.0499]], grad_fn=<UnsqueezeBackward0>)
tensor([[0.0541, 0.3523, 0.0342, 0.0682, 0.0275, 0.1247, 0.2284, 0.0193, 0.0386,
         0.0528],
        [0.0642, 0.0477, 0.0163, 0.0162, 0.0725, 0.0829, 0.2390, 0.2438, 0.1473,
         0.0701]], grad_fn=<DivBackward0>)
input tensor([[0.0541, 0.3523, 0.0342, 0.0682, 0.0275, 0.1247, 0.2284, 0.0193, 0.0386,
         0.0528],
        [0.0642, 0.0477, 0.0163, 0.0162, 0.0725, 0.0829, 0.2390, 0.2438, 0.1473,
         0.0701]], grad_fn=<DivBackward0>)
target tensor([9, 4])
range(0, 2)
2
 s3  tensor([0.0528, 0.0725], grad_fn=<IndexBackward0>)
 s4 negative :  tensor([-0.0528, -0.0725], grad_fn=<NegBackward0>)
 s5 log  tensor([2.9413, 2.6236], grad_fn=<NegBackward0>)
loss tensor(2.7825, grad_fn=<NegBackward0>)
```

<h4>Step 3</h4>
Ok, so we see `loss tensor(2.7825, grad_fn=<NegBackward0>)`. This is the value of this graph after performing this calculation and returning `cross_entropy`, which is equivalent to `F.cross_entropy` from PyTorch. Let's move to micrograd part. Starts from [29] listing. <br /><br />

But I won't describe it step by step here, because anyone can look at the .ipynb file and follow the log. The calculation results for these graphs are important here. I did it manually because these graphs are more transparent. So in listing [41] where `loss1` is executed, we see the result below `Value(data=23.68459169376059, grad=1)`. This is the first branch.

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/2c6b702fb9cdf74cec56e3e2d0e159709e7e8e92/7%20-%20micrograd%20exercises/3%20-%20exercise%20number%207-3/branch1.svg)

<br /><br />
Second branch. Lising [40] and result `Value(data=9.05013542436278, grad=0)`. And after this as we see `loss2.backward()` in listing [42] and then `draw_dot(loss2)` in [43].


![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/2c6b702fb9cdf74cec56e3e2d0e159709e7e8e92/7%20-%20micrograd%20exercises/3%20-%20exercise%20number%207-3/branch2.svg)

<br /><br />
<b>If we look up at that PyTorch log, we can see `sum(-1) tensor([23.6842, 9.0499], grad_fn=<SumBackward1>)` there.</b> These are the results for these 2 branches.
<br /><br />
And this is the first part that performs the `Softmax` function.
<br /><br />
If we add both values ​​together, the result is exactly `Value(data=32.734727118123374, grad=0)`, lisings [54], [55] . Then I call to `loss.backward()` and display graph in listings [56], [57]. And this is the result of combining both of these loss values ​​from each branch into one.


![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/2c6b702fb9cdf74cec56e3e2d0e159709e7e8e92/7%20-%20micrograd%20exercises/3%20-%20exercise%20number%207-3/softmax.svg)

This was the `Softmax` part -> `def softmax(x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)`. Because we first calculate `pred`, which are then used by `nl`, as can be seen below.

```
def softmax(x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
def nl(input, target): return -input[range(target.shape[0]), target].log().mean()

pred = softmax(x)
loss=nl(pred, target)
loss
```

<h4>Step 4</h4>

Ok, final part -> `def nl(input, target): return -input[range(target.shape[0]), target].log().mean()`
<br /><br />
So we get probs that start with `input tensor([[0.0541, 0.35... ` 

```
tensor([[0.0541, 0.3523, 0.0342, 0.0682, 0.0275, 0.1247, 0.2284, 0.0193, 0.0386,
         0.0528],
        [0.0642, 0.0477, 0.0163, 0.0162, 0.0725, 0.0829, 0.2390, 0.2438, 0.1473,
         0.0701]], grad_fn=<DivBackward0>)
input tensor([[0.0541, 0.3523, 0.0342, 0.0682, 0.0275, 0.1247, 0.2284, 0.0193, 0.0386,
         0.0528],
        [0.0642, 0.0477, 0.0163, 0.0162, 0.0725, 0.0829, 0.2390, 0.2438, 0.1473,
         0.0701]], grad_fn=<DivBackward0>)
target tensor([9, 4])
range(0, 2)
2
 s3  tensor([0.0528, 0.0725], grad_fn=<IndexBackward0>)
 s4 negative :  tensor([-0.0528, -0.0725], grad_fn=<NegBackward0>)
 s5 log  tensor([2.9413, 2.6236], grad_fn=<NegBackward0>)
loss tensor(2.7825, grad_fn=<NegBackward0>)
```

The classes are (hard coded) for example, as `target tensor([9, 4])`. You could choose any other numbers, but in this specific case, for this `xs`, they range from 0 to 9, totaling 10 classes. And it is easy to calculate only two classes by hand.
<br /><br />
So, `[9, 4]` are indices that correspond to, for example, embeddings that is, the vector for a given unique example. In this particular example, I am selecting 9 and 4.
<br /><br />
If we check this log again, we can see that for the first vector, index 9 contains `0.0528`. And the second vector, at position (index) 4, contains `0.0725`. `s3  tensor([0.0528, 0.0725], grad_fn=<IndexBackward0>)`

```
tensor([[0.0541, 0.3523, 0.0342, 0.0682, 0.0275, 0.1247, 0.2284, 0.0193, 0.0386,
         0.0528],
        [0.0642, 0.0477, 0.0163, 0.0162, 0.0725, 0.0829, 0.2390, 0.2438, 0.1473,
         0.0701]], grad_fn=<DivBackward0>)
```

btw. `one_hot` does something similar, it masks the other indices by setting them to 0, while setting the selected index to 1.
<br /><br />
Next, the .log() values ​​are calculated, as can be seen further on in `s5 log tensor([2.9413, 2.6236], grad_fn=<NegBackward0>)`
<br /><br />
This can be quickly verified by inputting these numbers as a tensor and applying .log(). `torch.tensor(0.0528).log(), torch.tensor(0.0725).log()` --> `(tensor(-2.9412), tensor(-2.6242))`. That is why a `-` is added, so that these values ​​become positives `-torch.tensor(0.0528).log(), -torch.tensor(0.0725).log()`.
<br /><br />
Last result, after .mean() `loss tensor(2.7825, grad_fn=<NegBackward0>)`
<br /><br />
<b>Pay attention to the outputs from the node m5a and n10a</b>. Before the .log() operations, we see 0.0528 for n10a and 0.0725 for m5a. Then, the .log() operation is applied, yielding results that are subsequently processed via .mean(), resulting in a value of 2.7825 for these two classes on this simple model (graph).

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/2c6b702fb9cdf74cec56e3e2d0e159709e7e8e92/7%20-%20micrograd%20exercises/3%20-%20exercise%20number%207-3/final_graph.svg)

<h4>Step 5</h4>

This is the same result as that calculated by the micrograd, for the entire `cross_entropy` graph, the value also comes out to 2.7825.
<br /><br />
The same as `F.cross_entropy(xs,ys)` --> `tensor(2.7825, grad_fn=<NllLossBackward0>)`

<h3>Summary</h3>
The goal is to build a simple "helper" for "Cross Entropy." This serves as an introduction to the exercises in *makemore* https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/makemore.py.
<br /><br />
There are slight discrepancies compared to what is shown in the graphs. This is because I entered numbers into micrograd with a precision of 3 decimal places for example, `n1 = Value(0.2477, label = 'n1')` whereas the original number had higher precision: `0.24770359694957733`. Hence these differences.
<br /><br />
The third graph, with the value 32.734727118123374 from listing [57], likely contains accumulated gradients. But that’s fine. Take a look at the final graph. The gradients are calculated correctly there, and `n1.grad` is 0.0270, just as it should be.
<br /><br />
https://github.com/KarolDuracz/Machine-Learning/blob/main/7%20-%20micrograd%20exercises/3%20-%20exercise%20number%207-3/Untitled2%20-%2019-10-2026%20-%20torchviz.ipynb - `torchviz` for comparison. You can see that the first graph, the "raw" version looks different, but it yields the same result. The original `loss = F.cross_entropy(xs, ys)` involves `LogSoftmaxBackward0` and `NllLossBackward0`.
<br /><br />
** Take another look at the numbers in the micrograd implementation that is, at n1, n2, ..., m1, m2... I made some typos, and some of the numbers are not exactly the same as those in `xs.data`. This is simply a copy-paste of these numbers from `x.data` into the containers of a `Value` object in micrograd.
<h3>References</h3>
1. https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/ - an interesting post about binary cross entropy.
<br /><br />
// 19-10-2026
