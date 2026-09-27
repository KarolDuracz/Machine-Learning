<h3>Helper - Visualizing functions and their derivatives</h3>

List of files: <br />
1. <b>slope_calculator_v2.html</b> - interactive web app to calculate the slope between two points. Look at the image below. <br />
2. <b>0_bYg0dna8SODDxieE.jpg</b> - (image) tanh formulas. <br />
3. <b>sigmoid_fn_derv.png</b> - (image ) sigmoid formulas. <br />
4. <b>Untitled - 27-09-2026 -graph of a function.ipynb</b> notebook file with code for this exercise.
<br /> 
<h3>Continuation for exercise 7-1 on the power rule</h3>
Look at listing [57] in this .ipynb file. If we take examples from the nodes I calculated in Exercise 7-1, we get exactly this graph and the results for xs[70], f_m_d(xs[70]) --> (tensor(2.), tensor(12.)). By overlying both graphs, the points for both functions become visible, much clearer why did I get a 12.0 for x00.grad in EX 7-1. 
<br />

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/7%20-%20micrograd%20exercises/2%20-%20exercise%20number%207-2/power%20rule%20plot.png?raw=true)

<b>EXPLANATION OF THE PLOT ABOVE</b><br />
Here, you can see two plots for the functions named `f_m` and `f_m_d` in my notebook (the `.ipynb` file). A vertical line is positioned at x = 2.0. This line intersects both functions at points obtained via `f_m_d(xs[70])` and `f_m(xs[70])`. Why `xs[70]`? Because that is the index corresponding to the value 2.0 on the x-axis specifically, the index resulting from the generation of `xs` using `torch.arange(-5, 5, 0.1)`. <br /><br />
So, what are the results for these two values, `f_m_d(xs[70])` and `f_m(xs[70])`? They are 2.0 and 8.0. That corresponds to the results for `n * X ** (n - 1) = 2.0` and `X ** 3 = 8`—in this case, `2 ** 3 = 8.0`.<br /><br />
That's it.
<br /><br />
If we examine the second point at 27.0, the lines of the function's graph approach each other and intersect. That is why both results—for x01 and x01f1—are 27.0.And this is clearly visible here.
<br /><br />
And what comes next in the web-based (HTML) version of the calculator? Here, we have the values ​​from node `x00` the starting point—after the first forward pass, which yields gradients for the value 2.0. Then, the next step (Step 2) updates `x00.data` to 1.88, and so on. Essentially, the difference is calculated with the optimizer such as SGD or AdamW disconnected. So, I’m calculating how much the point on this function changes visually its deltas, slope, etc. After the first and second steps in this case, for the next step I will take 1.88 and the subsequent value for step 3, and so on.

<h3>slope_calculator_v2.html</h3>

1. Field for entering the formula of any function. Here, the derivative for x ** 3. <br />
2. Enter 2 points. Refer to exercise 7-1 from the previous folder here. These are the points for the node I was calculating. The initial value is 2.0, followed by 1.88. Those are the two values.<br />
3. You can zoom in using the mouse wheel. <br />
4. For the result regarding these two points, see Exercise 7-1.

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/7%20-%20micrograd%20exercises/2%20-%20exercise%20number%207-2/slope%20calc%20helper_.png?raw=true)

// 27-09-2026
