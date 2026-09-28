<h3>Introduction - NLP (Natural Language Processing)</h3>
The plan is to start with something simple and break it down into its component parts. The exercises will utilize 
the makemore (https://github.com/karpathy/makemore/tree/master) implementation, involving an attempt to reproduce calculations and predictions comparing micrograd vs makemore, and so on.
<br /><br />
I’m not sufficiently prepared right now to tackle this NLP topic. Makemore and the names dataset train a simple statistical 
model that simply predicts the next letter or token using a simple equation for an MLP ( Multilayer perceptrons ) with formula

$\ tanh(\\vec{w}⋅\mathrm{X} + b)\$

This refers to the concept (https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) but I think it was sometime before 2017, before the Transformer architecture concept emerged (https://arxiv.org/pdf/1706.03762)
The people working on this problem concluded (I belive) that the goal shouldn't just be to statistically predict the next word, letter, or token in a sequence. 
A Transformer is capable of learning much more about text semantics, structure, and so on. It is a more complex model 
while it still learns statistical patterns much like an MLP but its greater complexity allows it to better capture the semantics and structure of the text from the data.
<br /><br />
But I’d like to do a few exercises using micrograd to better understand this architecture and the approach. However, that requires a better 
background in NLP itself and so on. Right now, I don't have enough skills to dive into that.
<br /><br />
// 27-09-2026 - This is the plan for the next exercises.

<br /><br />
<h3>A rough draft of thoughts...</h3>
Using simple terminology and analogies. The `tanh(xw+b)` non-linear activation function allows certain "active" neurons to pass through. By plotting this layer and its values ​​(e.g., using `plt.plot`), we can visualize the distribution of values ​​after they have passed through the activation function. This simple model can only learn the statistics of the relationship between a context (e.g., 3 characters) and the next character (or token). The situation changes, however, when we introduce a time-based component much like a Transformer does. Transformers possess channels that not only measure (or learn) patterns as an MLP does, but also capture the token's position within the sequence. What is the benefit of this? Theoretically, similar tokens can occupy similar positions within a given context. Consider, for instance, comparing C++ with C, or C with Python. Tokens located at position 11 in the text might differ in name, but because the model also learns about their position within the context, it gains more information about the token itself even if their names differ and the text's semantics are different.
<br /><br />
It's worth checking out in that way and  break it down into pieces in these exercises.
