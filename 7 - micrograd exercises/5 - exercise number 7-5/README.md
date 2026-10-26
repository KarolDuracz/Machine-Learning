// TODO - 26-10-2026
<br /><br />
Introduction
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
