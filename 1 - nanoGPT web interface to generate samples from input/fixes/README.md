<h3>Version 2</h3>
The folder contains app2.py and index2.html. Change the name to app.py and index.html and replace the previous files.
<br /><br />
This version has an additional mode -> token (stream) which performs responses to the web application after generating each token, each character. This means that the user does not have to wait for all predictions to be recalculated, but receives token after token after each processing by the network.
<br /><br />
And you can stop generating by pressing a button and then start generating subsequent distributions. So it's better to quickly get the answer and samples from the model character by character
<br /><br />

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/images/132%20-%2017-03-2026%20-%20token%20stream%20example%20ver%202.png?raw=true)
