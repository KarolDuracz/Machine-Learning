<h3>A list of tiny demos that allow to understand more complex code and model structures a little better</h3>

<h4>1. Linear XW + B - Ok, here's something to start with. Linear layer, forward pass only.</h4>

file: demo1.html

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/images/demo1%20-%20basics%20-%20forward%20pass%20xw%20b.png?raw=true)

What does the network actually learn from the data representation and what does it look for the parameter 
settings e.g. in this model https://github.com/karpathy/makemore/blob/master/makemore.py#L350 ? Without going into details now. Only forward pass helper as <b>demo1.html</b>
which illustrates the probability calculations in the table. But the <b>table only calculates the result for each row!</b>. This isn't a fully connected network where all 
neurons are multiplied together in linear layer. Here's just a simple demo that generates calculations for each row. The network we know today, for example, nn.Linear, 
which is used here in MLP, performs multiplication with all neurons in the next layer (fully connected). So, the calculations impact to all neurons that are connected.
<br /><br />
OK, that's it. It's just a table that has a 2D embedding vector for each letter based on [0.283, -0.232] plus some randomness. The decision is simply drawing a random number from the range 0-26. 
It doesn't matter, it's just a random number and highlighting. The MSE and CrossEntropy calculations don't matter either. It's just the result for each row.
<br /><br />
I'll write it again now. This is a calculation for each row. It's not a network. But it helps to understand the X W + B calculations. And basic forward pass operations.
<br /><br />
Press F5 to generate other outputs
<hr>
<h4>2. Features extraction and Classification</h4>

There are many similar ones, but I like this post https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/#entry-content-comments .
<br /><br />
Good article, that described in a simple way the fundamental concepts of networks. And a very good image shows something important: the feature extraction phase, what the model looks for in the data. And then classification and sampling.
<br /><br />
One word to Andrej's nanoGPT ( LLM concept ) https://github.com/karpathy/nanoGPT . nanoGPT is scalable because you can change several parameters easily to increase the size of the model and it generalizes quite well. And this model has really good enough "features extraction", to build base model. 
