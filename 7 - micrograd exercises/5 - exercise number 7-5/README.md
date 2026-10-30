<h4>Demo 1 - Warmup</h4>

File ` Untitled - 29-10-2026 - 7-5 ex warmup.ipynb ` . For now I won’t go into more detail  why I included this specific code. The most important part begins in listings [36]–[52], where I encounter a simple problem, using an MSE (Mean Squared Error) function, the parameters are adjusted to improve (minimize) the loss value. This is illustrated in [46], where the output `xs` is shown to be virtually identical to the `targets`. Moving on to listings [53]–[60], starting at [53] marked with the comment `# more complex NN model` we see a "model" that produces 10 values ​​for each index (let's say token) (a 10 element vector for every input). There are three such inputs. The process remains the same, in listing [59], we generate predictions using `.multinomial`. The `multinomial` function works by taking the values ​​and generating samples based on their relative frequencies (with higher values ​​having a higher probability of being sampled). You could observe this, for instance, by generating a larger number of samples let's say 100 and analyzing the resulting distribution plot. But that is a topic for another time.
<br /><br />
<b>What is crucial to understand is that a neural network requires a problem defined by a mathematical equation. We need to define the problem.</b> Then the network will find a solution
<br /><br />
Let me return for a moment to Andrej’s YouTube videos. If we listen closely to what he says specifically his suggestion to "try to find a lower loss"  ( better settings, parameters to beat the loss ) why is this so significant, for instance in an MLP model used for predicting the next letter or token? Because if we continue with this exercise (warmup), we will come to understand that WITHOUT IT, THE NETWORK WILL NOT PROPERLY LEARN THE DATA and will end up generating GARBAGE!
<br /><br />
In other words, we can construct various functions and measure loss, but without a defined problem, this serves no useful purpose.
<br /><br />
<i>One more thing. What is shown in listing [27] describes a type of "compute unit" that has 1 input value, an activation function, and 1 output. Listing [37] describes the same "neuron," but with 3 inputs, an activation function, and 3 outputs. In [54], there are 10 weights, one for each input, activation function and 3 outputs. </i>  Photo for comparison https://upload.wikimedia.org/wikipedia/commons/8/8c/Perceptron_moj.png?utm_source=pl.wikipedia.org&utm_campaign=imageinfo&utm_content=thumbnail_unscaled
<br /><br />
<i>I refer to Andrej’s YouTube recordings, and suggest looking through the initial exercises on building an MLP from scratch https://www.youtube.com/@AndrejKarpathy/videos . To the backprop ninja video. So, when I wrote about "finishing" this "warmup" above on demo 1, I mean building something that functions like `makemore` using simple names or something like tiny Shakespeare. That’s not my actual goal, though I’ve already gone through those exercises. Just something to keep in mind. I’m just writing this in case someone is here for the first time and doesn’t know this. I’ve already built simple models using names.txt as exercises, so I’m aware of that. I’m already looking into larger models and networks, and data. So I'm moving forward...</i>
<br /><br />
That's all for now.
<br /><br />
// 30-10-2026
<hr>
<br /><br /><br />
// TODO - 26-10-2026
<br /><br />
<b>Introduction</b>
<br /><br />
Andrej covered an exercise on LeNet (https://github.com/karpathy/lecun1989-repro), but I feel it’s missing a look at a few aspects, for instance, the "filters," which act 
as an intermediate layer (or a form of activation), that breaks down the input data into smaller components, allowing the model to learn features through various filters.
And layer by layer, it learns these "features" as a data concept. And as a larger network composed of these small parts, it performs calculations for the entire graph. 
<br /><br />
But that’s a LeNet and MNIST problem. But what about LLMs, transformer architecture? Following this way  of reasoning, a small MLP model structured as linear -> tanh -> linear is likely 
unable to learn much in terms of these "features" as it lacks a sufficient number of filters beyond the tanh activation.
<br /><br />
And so on... Stuff like that. So the question is: what exactly does a transformer learn, and what kind of "feature extraction" is performed by its layers? 
Is there any way to verify this?
<br /><br />
But that’s all for now. An initial idea worth returning to. WHY DO I EVEN THINK THIS? Lately, the term "scale" has been coming up in the conversations about of these chatbots, LLMs, and so on. And, in truth, it made sense looking at the 
initial tests of models with 124M, 774M, 1.5B parameters, and so on. The plots with some benchmarks showed that there was potential and a model with a larger number of parameters provides better answers, and that it was worth making the models larger. To add more parameters. But...
<br /><br />
Perhaps it is worth returning to those original concepts for a moment and taking a closer look at them.
<br /><br />
<h4>References</h4>
1. http://karpathy.github.io/2022/03/14/lecun1989/
