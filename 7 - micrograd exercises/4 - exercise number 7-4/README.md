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

<br /><br />

<h4>Okay, it’s time to do the same exercise for 3 steps using `cross entropy`. </h4>

1. So, once again, starting with the PyTorch version, we have the initialization in [27], followed by the forward pass in [29]. In [32], I display the loss. After step 1 the value is `tensor(2.7825, grad_fn=<NegBackward0>)`
2. Next, micrograd from [33]; [34] covers initialization, [35] the forward pass, [36] the loss, and [37] generating the graph to a file.
3. The same steps as before.

<b>Step 1.</b>

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/26cef26c6e4d2e3f3f72c6c37814e0843bdb8ae2/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_ce_step1.svg)

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

`xs.data` is now in [30]!

```
tensor([[ 0.2423,  2.0864, -0.2151,  0.4721, -0.4318,  1.0704,  1.6652, -0.7835,
         -0.0941,  0.3182],
        [-0.5498, -0.8439, -1.9134, -1.9239, -0.3281, -0.2957,  0.7476,  0.7668,
          0.2727, -0.4623]])
```

<b>Step 2</b>

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/26cef26c6e4d2e3f3f72c6c37814e0843bdb8ae2/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_ce_step2.svg)

```
raw  tensor([[ 0.2423,  2.0864, -0.2151,  0.4721, -0.4318,  1.0704,  1.6652, -0.7835,
         -0.0941,  0.3182],
        [-0.5498, -0.8439, -1.9134, -1.9239, -0.3281, -0.2957,  0.7476,  0.7668,
          0.2727, -0.4623]], requires_grad=True)
 exp  tensor([[1.2742, 8.0556, 0.8065, 1.6033, 0.6494, 2.9166, 5.2868, 0.4568, 0.9102,
         1.3747],
        [0.5771, 0.4300, 0.1476, 0.1460, 0.7203, 0.7440, 2.1119, 2.1529, 1.3135,
         0.6298]], grad_fn=<ExpBackward0>)
 sum  tensor(32.3071, grad_fn=<SumBackward0>)
 sum(-1)  tensor([23.3340,  8.9731], grad_fn=<SumBackward1>)
tensor([[23.3340],
        [ 8.9731]], grad_fn=<UnsqueezeBackward0>)
tensor([[0.0546, 0.3452, 0.0346, 0.0687, 0.0278, 0.1250, 0.2266, 0.0196, 0.0390,
         0.0589],
        [0.0643, 0.0479, 0.0164, 0.0163, 0.0803, 0.0829, 0.2354, 0.2399, 0.1464,
         0.0702]], grad_fn=<DivBackward0>)
input tensor([[0.0546, 0.3452, 0.0346, 0.0687, 0.0278, 0.1250, 0.2266, 0.0196, 0.0390,
         0.0589],
        [0.0643, 0.0479, 0.0164, 0.0163, 0.0803, 0.0829, 0.2354, 0.2399, 0.1464,
         0.0702]], grad_fn=<DivBackward0>)
target tensor([9, 4])
range(0, 2)
2
 s3  tensor([0.0589, 0.0803], grad_fn=<IndexBackward0>)
 s4 negative :  tensor([-0.0589, -0.0803], grad_fn=<NegBackward0>)
 s5 log  tensor([2.8317, 2.5224], grad_fn=<NegBackward0>)
loss tensor(2.6770, grad_fn=<NegBackward0>)
```

`xs.data` after step 2

```
tensor([[ 0.2368,  2.0518, -0.2185,  0.4652, -0.4346,  1.0579,  1.6426, -0.7855,
         -0.0980,  0.4123],
        [-0.5562, -0.8487, -1.9150, -1.9256, -0.2362, -0.3040,  0.7241,  0.7428,
          0.2581, -0.4694]])
```

<b>Step 3</b>

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/26cef26c6e4d2e3f3f72c6c37814e0843bdb8ae2/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_ce_step3.svg)

```
 raw  tensor([[ 0.2368,  2.0518, -0.2185,  0.4652, -0.4346,  1.0579,  1.6426, -0.7855,
         -0.0980,  0.4123],
        [-0.5562, -0.8487, -1.9150, -1.9256, -0.2362, -0.3040,  0.7241,  0.7428,
          0.2581, -0.4694]], requires_grad=True)
 exp  tensor([[1.2672, 7.7822, 0.8037, 1.5923, 0.6475, 2.8804, 5.1684, 0.4559, 0.9067,
         1.5103],
        [0.5734, 0.4280, 0.1473, 0.1458, 0.7896, 0.7379, 2.0628, 2.1018, 1.2944,
         0.6254]], grad_fn=<ExpBackward0>)
 sum  tensor(31.9211, grad_fn=<SumBackward0>)
 sum(-1)  tensor([23.0147,  8.9064], grad_fn=<SumBackward1>)
tensor([[23.0147],
        [ 8.9064]], grad_fn=<UnsqueezeBackward0>)
tensor([[0.0551, 0.3381, 0.0349, 0.0692, 0.0281, 0.1252, 0.2246, 0.0198, 0.0394,
         0.0656],
        [0.0644, 0.0481, 0.0165, 0.0164, 0.0887, 0.0828, 0.2316, 0.2360, 0.1453,
         0.0702]], grad_fn=<DivBackward0>)
input tensor([[0.0551, 0.3381, 0.0349, 0.0692, 0.0281, 0.1252, 0.2246, 0.0198, 0.0394,
         0.0656],
        [0.0644, 0.0481, 0.0165, 0.0164, 0.0887, 0.0828, 0.2316, 0.2360, 0.1453,
         0.0702]], grad_fn=<DivBackward0>)
target tensor([9, 4])
range(0, 2)
2
 s3  tensor([0.0656, 0.0887], grad_fn=<IndexBackward0>)
 s4 negative :  tensor([-0.0656, -0.0887], grad_fn=<NegBackward0>)
 s5 log  tensor([2.7238, 2.4230], grad_fn=<NegBackward0>)
loss tensor(2.5734, grad_fn=<NegBackward0>)
```

`xs.data`

```
tensor([[ 0.2313,  2.0180, -0.2220,  0.4583, -0.4374,  1.0454,  1.6201, -0.7875,
         -0.1019,  0.5058],
        [-0.5626, -0.8535, -1.9167, -1.9272, -0.1450, -0.3123,  0.7009,  0.7192,
          0.2435, -0.4764]])
```

<b>extra Step 4 - only graph</b> - Look at `0.2313` between `n1` and `n1a`

![dump](https://raw.githubusercontent.com/KarolDuracz/Machine-Learning/7fb59ce67f2139cc98806fef94a6ad69b3ac457d/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/graph_ce_step4.svg)

<h4>Summary</h4>

Okay, let’s trace the CrossEntropy version again. After step 1, the log shows `raw tensor([[ 0.2477, 2.1216,...` at the top these are the values ​​for `n1` and `n2`. Further down, we see `xs.data is now in [30]!`, which contains the subsequent values the ones appearing between `n1` and `n1a` in the micrograd graph: `tensor([[ 0.2423, 2.0864,`. Now for STEP 2: in `raw tensor([[ 0.2423, 2.0864`, we can see the value at index 0 is `0.2423`. Then comes `xs.data after step 2`, followed by `tensor([[ 0.2368, 2.0518, -0.2185,...`. Indeed, on the graph after step 3 where there are two columns between `n1` and `n1a` we can see these two values to the right of `n1`, there is `0.2423`, followed by `0.2368`, and then `n1a` with a value of `1.2672`.
<br /><br />
An additional graph after step 4, as we can see above in `xs.data`, the value `0.2313` should appear.
<br /><br />

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/7%20-%20micrograd%20exercises/4%20-%20exercise%20number%207-4/step4_lookat.png?raw=true)

<br />
The goal is to understand how to interpret this and whether it’s actually useful, or how to use it correctly for analysis.
<br /><br />
// 22-10-2026
