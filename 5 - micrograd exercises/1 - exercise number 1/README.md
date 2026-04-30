<h3>Description</h3>

Here are the notebooks for the "micrograd" video - https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd
<br /><br />
In file https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb we can find this example of building graphs line by line until the .backward() call. ( Ok, there are more examples of this type in this notebook, but this one shows how to build a graph in this way, which is displayed as an image ).

```
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
# ----
e = (2*n).exp()
o = (e - 1) / (e + 1)
# ----
o.label = 'o'
o.backward()
draw_dot(o)
```

I used this approach. Then it can be compare it with the PyTorch implementation, which has a supporting graph library. It's better than nothing: https://pypi.org/project/torchviz/

<h3>Now take a look at the file I posted here - a description of what exactly is here</h3>

Open file from this folder - ``` 1_exercise_1_micrograd.ipynb ```

<h4>step by step what is it about:</h4>
1. First, install torchviz.<br />
2. I've included the Value code, which can be found in micrograd_lecture_second_half_roughly.ipynb. <br />
3. Added support to Value for log, exp, and sigmoid. <br />
4. Next, a quick test to see if it renders graphs for - d = a * b + c. Okay, it works. <br />
5. Graphviz part. <br />
6. And then there's DEMO 1 - a line-by-line implementation, and then below that, pytorch, which generates a graph - a simple linear operation z0 = x @ w + b. <br />
7. Then there's DEMO 2 - the same thing, only after XW+B there's sigmoid. And that's it. And there are graphs for that too. <br />
8. There are also explore functions that look into the pytroch computational graph and print the steps for the graphs. <br />
<br /><br />
Mikrograd - demo 1
<br /><br />

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/7c4addcf3a976d372057d6aba219ebeeaded2323/5%20-%20micrograd%20exercises/1%20-%20exercise%20number%201/micrograd%20graph%20demo1.svg)

Trochviz - demo 1
<br /><br />

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/7c4addcf3a976d372057d6aba219ebeeaded2323/5%20-%20micrograd%20exercises/1%20-%20exercise%20number%201/pytorch%20torchviz%20demo1.svg)

<br />
This can help analyze both micrograd and graphviz graphs. This is more visible in demo2 in the .ipynb file, where sigmoid is added. This can help to some extent. Therefore, this is my setup for these exercises here.
<br /><br />
// 30-04-2026
