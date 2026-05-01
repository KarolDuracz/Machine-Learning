<h3>Description</h3>

In the first exercise I only described the initial setup and an idea of ​​what I'll be doing. Here's a specific demo built to simulate backward pass and parameter updates manually. Andrej did it. But to do it again with graphs here.
<br /><br />
FILE : <b>exercise_1_demo_ex_2_0105-2026_.ipynb</b>
<br /><br />
The first part from the top is PyTorch<br />
1. setup parameters<br />
2. setup learning rate, target, loss function<br />
3. forward pass, backward, and update params<br />
4. torchviz graph<br />
5. explore

Micrograd with Value.
<br />
1. setup<br />
2. forward pass, backward pass, and update params w0, w1, b<br />

Loss values ​​are the same, after 1x forwad pass<br />
micrograd loss = 0.4761 | torch loss = 0.4761
<br /><br />
Now clicking on forward pass in the listings -> In [189] - forward pass for pytorch implementation and In [196] - forward pass for micrograd. Or by clicking forward a second time, loss is decreasing 0.2195, parameters are changed and the same for micrograd.
<br /><br />
<b>That's it. This setup is meant to be here as an example of how to manually do forward pass, backward, and update params.</b>
<br /><br />
// 01-05-2026
