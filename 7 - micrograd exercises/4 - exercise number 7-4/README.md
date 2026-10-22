<h3>A few notes about the graphs generated in these micrograd exercises, notes for exercise 7-3</h3>

1. First of all. This notebook, `Untitled - ex 7-4 - 22-10-2026.ipynb`, is "clean" meaning the code has been executed only once. This ensures that the sequence from 1 to 37 remains in the correct order.
2. I added a few `print` statements and a couple of lines of code, but there’s nothing here worth commenting on. I am writing about listing 1-4, which consists of the micrograd + graphviz code.
3. I will return briefly to exercise 7-1. It begins at listing [5] with the comment `# Perform the test from exercise 7-1 again.`. I then execute this code twice, resulting in a `loss data` value of `35.0000` for the first test, this corresponds to listing [7]. Next, in listing [8], I perform a second step, which yields a `loss data` value of `26.9911`. In essence, I am simply re-running exercise 7-1 for two steps in order to generate the graphs shown in the listing, and to observe via `print` statements, what is happening under the hood.
4. Okay, now let’s look at listing [8] and the graph generated after 2 steps. The loss is 26.9911. The yellow boxes highlight unique nodes from the original graph (from the first run). As we take further steps between nodes for example, between `x01f1` and `x01` a white node appears, containing the value `data 2.7300`.
5. If I run the forward pass again, I will get a loss of 21.3282, and between nodes `x01` and `x01f1`, I will see two new, white nodes.
6. This simple example (7-1 ex) helps in understanding the process of building a graph and analyzing it.
<br /><br />
<b>Test 1</b>
<br /><br />
7. <b>Okay, let’s get to the main point. </b>Listing [9] begins with the comment `# second test`. First, I modify the `trace`, `vis_color`, and `draw_dot` functions by removing the `print` statements, see Listing [10]. I also change the final line to `dot.render(outfile, format=format)`, which saves the output to a file on the disk instead of rendering it within the notebook. I forgot to add `return` specifically `return dot.render(outfile, format=format)` but it executes anyway. Simply removing junk code.
8. Ok, listing [11] `# micrograd branch 1`. Just as I did in exercise 7-3, I didn’t change any numbers. I simply copy-pasted the same content from the previous exercise to get the same results.
9. Listing [12] contains the initialization. This is followed by [13], which includes the forward pass, backward pass, and an update using a learning rate of `LR=0.2`. In listing [14], I execute `draw_dot(loss1)`, but the graph is saved to a file on the local disk.

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/073264e91719333298e66a9e932a58115ad2bee9/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_test1.svg)


10. Next, there is a version using PyTorch, similar to the one in exercise 7-3 (Listing [15]). First, in Listing [16], I add a loss function that performs only an `exp()` operation. I’ve included `print` statements so we can see the results.
11. Next, in [18], there is an initialization using the same seed, `3409`, as in 7-3.
12. Next, after initializing `xs.data`, we proceed to the forward pass, backward pass, and update step with `LR=0.2`, following the same approach as in the micrograd example above as shown in listing [21].
13. Then, in [22] and [23], I look at `xs.data` and `xs.grad` once again, this time after the backward pass and update. Alright, I'll back to graph from `micrograd` shown above, and log for the pytorch version forward pass. <b>You need to look at the graph and the line `after exp()`</b>

```
 after exp() 
tensor([[1.2811, 8.3445, 0.8093, 1.6143, 0.6511, 2.9532, 5.4089, 0.4577, 0.9...
```

14. `n1` 0.2477 leads to `n1a` 1.2811. This corresponds to index 0 of this array (vector). `n2` leads to `n2a`, where `n2a` contains 8.3445 following an exp() operation. And so on.
15. I am now describing listing [21], the PyTorch version and its log and comparing it with the graph from listing [14], `draw_dot(loss1)`, which is the image shown above.
<br /><br />
<b>Test 2</b>
<br /><br />
1. I do the same thing, but for the cross entropy version from 7-3 ex. It starts with listing [24], with the comment `# full graph for cross entropy`
2. `softmax` and `nl` versions without skipping -1, it is not needed here.
3. First, run the PyTorch version starting from cell [26]. Proceed step by step: initialization in [27]. Display `xs.data` after initialization to allow for comparison with the SVG graph visualization. Next, in [29], perform the forward pass, backward pass, and update. This will generate logs. Finally, in [30] and [31], examine `xs.data` and `xs.grad` after the backward pass and update have been completed.
4. Micrograd version now. Listing [33]. And again, initialization in [34]. This includes the errors I made in ex 7-3 specifically, incomplete numbers, e.g., `m4 = Value(-1.922, label = 'm4')`. Listing [35] contains the full code for the forward pass, backward pass, and update step. At the end, there is `last_loss` in [36], followed by saving the graph to a file in [37] using `draw_dot(last_loss)`.

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/073264e91719333298e66a9e932a58115ad2bee9/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_full.svg)

<br /><br />

<h4>Now, the main part, where I will perform a forward pass for a few steps, let's say, 2 or 3 to trace what emerges between the connections, much like the process of taking a few steps in the simple example from Ex 7-1. Power rule demo. </h4>
First of all, that’s why I loaded a clean notebook and didn’t execute the code immediately, instead I modified it to save the SVG file to disk, so I could get a clear picture of what was going on. Now I will perform a forward pass (in the cells containing the forward pass code) on local computer, and simply provide the logs here, along with a description and an SVG image.
<br /><br />

1. I’ll start with listing [12], which shows the micrograd version for `branch 1`, `loss`, and `Value(data=23.68459169376059, grad=1)` . Now I will generate a graph and save it to disk by executing `draw_dot(loss1)`.
2. Next, perform the same steps for PyTorch, as shown in listing [18] - [23].
3. As I wrote above, the logs from [21] (PyTorch version) now display the values ​​after `exp()`, showing: `tensor([[1.2811, 8.3445, 0.8093, 1.6143, 0.6511,...`. These correspond, in order, to `n1a`, `n2a`, `n3a`, etc.
4. Next `raw` , the values ​​shown above, starting from `tensor([[ 0.2477,  2.1216, -0.2116,  0.4789, -0.4290,  1.0829,...` correspond to `n1`, `n2`, and so on.

```
raw
tensor([[ 0.2477,  2.1216, -0.2116,  0.4789, -0.4290,  1.0829,  1.6881, -0.7816,
         -0.0902,  0.2235],
        [-0.5434, -0.8391, -1.9117, -1.9223, -0.4209, -0.2874,  0.7715,  0.7912,
          0.2874, -0.4553]], requires_grad=True)
 after exp() 
tensor([[1.2811, 8.3445, 0.8093, 1.6143, 0.6511, 2.9532, 5.4089, 0.4577, 0.9137,
         1.2504],
        [0.5808, 0.4321, 0.1478, 0.1463, 0.6565, 0.7502, 2.1630, 2.2060, 1.3330,
         0.6342]], grad_fn=<ExpBackward0>)
tensor([23.6842,  9.0499], grad_fn=<SumBackward1>)
```

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/508ba2655a07736c5627db3ba12f1bbf4df11be5/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_branch1_step1.svg)

5. In listing [23], we can see the values ​​that will reside in the intermediate nodes, for instance, the value `-0.0085` at index 0 will be located between `n1` and `n1a`.

```
tensor([[-0.0085,  0.4527, -0.3735,  0.1560, -0.5593,  0.4923,  0.6063, -0.8731,
         -0.2730, -0.0266],
        [-0.5434, -0.8391, -1.9117, -1.9223, -0.4209, -0.2874,  0.7715,  0.7912,
          0.2874, -0.4553]])
```

6. Let's check. First, the PyTorch [21] forward pass to get the loss (step 2). The result is `tensor([10.6150, 9.0499], grad_fn=<SumBackward1>)`.
7. Then go to [13] and do forward, and [14] to get graph. ( micrograd )

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/47966d9cb3ab004fc8baeaee1054c4b0548139da/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_branch1_step2.svg)

8. Between the yellow `n1` on the left and `n1a`, there is a new white node that contains `-0.0085`.
9. Okay, next step (step 3). Same thing. First, the forward pass for the PyTorch version. Cell [21].
10. But don't execute [23] yet, where `xs.data` is. Those values ​​should be there.

```
tensor([[-0.2068,  0.1382, -0.5112, -0.0777, -0.6736,  0.1651,  0.2396, -0.9566,
         -0.4252, -0.2213],
        [-0.5434, -0.8391, -1.9117, -1.9223, -0.4209, -0.2874,  0.7715,  0.7912,
          0.2874, -0.4553]])
```

11. Step 3 ( pytorch ) loss `tensor([8.2856, 9.0499], grad_fn=<SumBackward1>)` and logs

```
raw
tensor([[-0.2068,  0.1382, -0.5112, -0.0777, -0.6736,  0.1651,  0.2396, -0.9566,
         -0.4252, -0.2213],
        [-0.5434, -0.8391, -1.9117, -1.9223, -0.4209, -0.2874,  0.7715,  0.7912,
          0.2874, -0.4553]], requires_grad=True)
 after exp() 
tensor([[0.8132, 1.1482, 0.5998, 0.9252, 0.5099, 1.1795, 1.2707, 0.3842, 0.6536,
         0.8014],
        [0.5808, 0.4321, 0.1478, 0.1463, 0.6565, 0.7502, 2.1630, 2.2060, 1.3330,
         0.6342]], grad_fn=<ExpBackward0>)
tensor([8.2856, 9.0499], grad_fn=<SumBackward1>)
```

12. Now, the same steps for micrograd—that is, once again, the forward pass [13] and [14].

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/d98de853c2450cf03b932460edac1d5b682a37ad/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_branch1_step3.svg)

13. After `step 3`, we can see two columns between `n1` and `n1a`, starting from `n1` on the left, there is `data -0.0085`, followed by `data -0.2068`, and then `n1a` with `data 0.8132`. Once again, the intermediate values ​​appear in `xs.data` after the forward pass in the PyTorch version. In other words, by looking one step ahead, you can see what the intermediate values ​​in `micrograd` will be. <b>This might be valuable information.</b> ?!
