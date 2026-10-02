<h3>Notes, reflections, plans</h3>
1 - 02-10-2026 - I constantly have MNIST, LeNet, and AlexNet in the back of my mind. LLMs involve a somewhat different type of task, architecture, and data. 
What interesting things were demonstrated by Yann LeCun’s LeNet, and subsequently by AlexNet which in my view is a somewhat larger model but very similar to LeNet, <b>that’s a generalization.</b> However, it seems 
that AlexNet and the associated approach offer a more comprehensive take on this issue. Interestingly, they appear to have aimed at building a large, diverse dataset so the network could be trained on a wide 
variety of image variants. Essentially, feeding the model a vast array of diverse images gives the network or base model a broader pattern-recognition capability. This is certainly worth a closer look. 
The idea involves using highly diverse data to 
train a general base model that generalizes well and retains a multitude of pattern combinations. In my opinion, AlexNet advanced this approach somewhat, and this concept could potentially be applied to LLMs as well.
<br /><br />
In simple words: look at what was done in AlexNet and at the training set. Consider, too, where the LeNet and AlexNet networks get the ability to memorize a large number of patterns from that dataset. Where does that capacity come from?
<br /><br />
An LLM takes a slightly different approach because the Transformer attempts to predict the next token (word). I like the Ilya's analogy used in that interview (Fireside Chat With Ilya Sutskever and Jensen Huang: AI Today and Vision of the Future, March 2023). They are probably right about the analogy to crime novels and the "who committed the crime" question as a model task. And making increasingly accurate predictions implies that the model understands the subject matter of the text. It also seems to influence the model's IQ. However, the knowledge embedded in the network's internal connections isn't the whole story... But that approach really makes sense.
