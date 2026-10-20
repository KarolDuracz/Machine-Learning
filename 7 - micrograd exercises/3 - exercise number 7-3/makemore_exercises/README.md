<h3>Warm up with Makemore</h3>

1. Open the file `Untitled8 - 20-10-2026 - graph transformer torchviz and loss with ignore_index.ipynb`, which is located here.
2. Next, take a look at listing [207], which contains the beginning of a test for the revised `nl` function a version that ignores indices where the target value is `-1`. How did those `-1` end up there? This is a concept adopted by default in Andrej’s *makemore* , essentially a form of masking. We could dive into analyzing the code, but that’s not the goal right now.
3. The API provides a way to ignore specific elements via the `ignore_index` attribute in subsequent functions, which effectively masks (ignores) those tensors https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html or https://docs.pytorch.org/docs/2.14/generated/torch.nn.NLLLoss.html
4. Okay, let’s go with the version that includes the print. You see a list here in the middle that extracts elements into `arr`. PyTorch handles this differently, but the point is simply to check the loss value. Why did I get `tensor(3.3278, grad_fn=<NllLossBackward0>) tensor(3.0073, grad_fn=<NegBackward0>) tensor(3.3278, grad_fn=<NllLossBackward0>)` that is, the results from executing `print(loss_test, loss, final_loss)` within the MLP training loop mentioned above? Why 3.0073? Not 3.3278. Because the raw version does not ignore those -1 indices. Now it's fixed in some way.

```
def softmax(x): 
    s1 = x
    #print( " raw " , s1)
    s2 = x.exp()
    #print(" exp " , s2)
    s3 = x.exp().sum()
    #print(" sum " , s3)
    s4 = x.exp().sum(-1)
    #print(" sum(-1) " , s4)
    s5 = (x.exp().sum(-1)).unsqueeze(-1)
    #print(s5)
    s6 = x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
    #print(s6)
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

arr = []

def nl(input, target): 
    print("input", input)
    print("target", target)

    s1 = range(target.shape[0])
    s2 = target.shape[0]
    print(" s1 ", s1)
    print(" s2 ", s2)
    print(" range ", [range(target.shape[0])])
    print(" test1 : ", input[0])
    #print(" test2 : ", input[(0,15),0])

    global arr
    col_idx = 0
    for t in target:
        print(t)
        if (t == -1):
            print(col_idx, "negative")
        elif t != -1:
            print(col_idx, " to arr ")
            arr.append(-input[col_idx].log().mean())
        col_idx += 1

    s3 = input[range(target.shape[0]), target]
    print( " s3 " , s3)

    s4 = -input[range(target.shape[0]), target]
    print( " s4 negative : ", s4)

    s5 = -input[range(target.shape[0]), target].log()
    print( " s5 log ", s5)

    p_loss = -input[range(target.shape[0]), target].log().mean()
    print( "loss" , p_loss)
    
    return -input[range(target.shape[0]), target].log().mean()
    #return arr
```

5. I saved the logits `LOGITS_GLOBAL` and targets `TARGET_GLOBAL` from the training loop. See the makemore code. Listing [209]

```
pred = softmax(LOGITS_GLOBAL)
loss=nl(pred, TARGET_GLOBAL.view(-1))
```

6. This still returning `tensor(3.0073, grad_fn=<NegBackward0>)`, as seen further down at [210]. But... Let’s move on to the next cells (listings).
7. In [211], it displays values ​​from selected tensors

```
[tensor(2.7172, grad_fn=<NegBackward0>),
 tensor(3.0128, grad_fn=<NegBackward0>),
 tensor(3.9272, grad_fn=<NegBackward0>),
 tensor(4.2643, grad_fn=<NegBackward0>),
 tensor(2.9752, grad_fn=<NegBackward0>),
 tensor(3.8878, grad_fn=<NegBackward0>),
 tensor(2.8103, grad_fn=<NegBackward0>),
 tensor(3.0280, grad_fn=<NegBackward0>)]
```
8. We need to go back to the results from `makemore` to see what the values ​​are in the line that begins with `s5 log  tensor([2.7172, 3.0...`

```
target tensor([26,  8,  1, 22,  9,  1,  8,  0, -1, -1, -1, -1, -1, -1, -1, -1])
range(0, 16)
16
 s3  tensor([0.0661, 0.0492, 0.0197, 0.0141, 0.0510, 0.0205, 0.0602, 0.0484, 0.0679,
        0.0578, 0.0751, 0.0774, 0.0517, 0.0638, 0.0968, 0.0635],
       grad_fn=<IndexBackward0>)
 s4 negative :  tensor([-0.0661, -0.0492, -0.0197, -0.0141, -0.0510, -0.0205, -0.0602, -0.0484,
        -0.0679, -0.0578, -0.0751, -0.0774, -0.0517, -0.0638, -0.0968, -0.0635],
       grad_fn=<NegBackward0>)
 s5 log  tensor([2.7172, 3.0128, 3.9272, 4.2643, 2.9752, 3.8878, 2.8103, 3.0280, 2.6897,
        2.8511, 2.5887, 2.5591, 2.9620, 2.7516, 2.3351, 2.7571],
       grad_fn=<NegBackward0>)
loss tensor(3.0073, grad_fn=<NegBackward0>)
tensor(3.3278, grad_fn=<NllLossBackward0>) tensor(3.0073, grad_fn=<NegBackward0>) tensor(3.3278, grad_fn=<NllLossBackward0>)
```

9. So, this line. And these are the 8 extracted indices, from which .mean() is subsequently calculated.

```
 s5 log  tensor([2.7172, 3.0128, 3.9272, 4.2643, 2.9752, 3.8878, 2.8103, 3.0280
```

10. In listing [213] the result is 3.3278456032276154. Now it's correct.

<h3>Summary</h3>
Is this the correct solution? Probably not. This is a temporary solution for simple tests!
<br /><br />
Regarding this .ipynb file and the code in `makemore`: I’ve reduced the number of layers and parameters. I also set `num_workers=0` in the `batch_loader` to avoid runtime errors, changed `batch_size` from `32` to `1`, only 2 steps in the loop, as no more are needed for testing here, and so on. So now this model looks like this.
<br /><br />

```
MLP(
  (wte): Embedding(28, 2)
  (mlp): Sequential(
    (0): Linear(in_features=32, out_features=2, bias=True)
    (1): Tanh()
    (2): Linear(in_features=2, out_features=27, bias=True)
  )
)
```

There are probably a few other things, but that’s not important right now. I’ll come back to it during the exercises with makemore. At the bottom are the graphs for MLP and transformer.
<br /><br />
I’ve reduced the `batch_size` to `1`. Using an API that loads mini-batches for instance, with a size of `32` tends to yield significantly better results after, let's say, `10,000` training steps compared to using a `batch_size=1`. The resulting plots and the samples themselves look different because the model encounters the data more frequently within a given context. But that’s a topic for another time. For the purpose of these tests, it’s better to set it to `1`.
<br /><br />
The goal was to examine these differences in loss values: `tensor(3.3278, grad_fn=<NllLossBackward0>) tensor(3.0073, grad_fn=<NegBackward0>) tensor(3.3278, grad_fn=<NllLossBackward0>)`
<br /><br />
// 20-10-2026
