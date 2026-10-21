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

