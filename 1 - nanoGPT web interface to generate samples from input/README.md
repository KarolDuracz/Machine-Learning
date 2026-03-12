<h3>Introduction</h3>
MLP from 2003, and subsequent evolutions, including Transformer from 2017, i.e. Generative Pre-trained Transformer, have an interesting property - i.e. recognizing patterns in text based on what is in the input, i.e. context.
<br /><br />
This demo creates a web interface that communicates with the sample.py script from the repo https://github.com/karpathy/nanoGPT . Instead of using the CMD console with sample.py script, user can send queries from the web. Secondly, what interests me most is generating N samples for a given input (context), e.g., the phrase "To be, or not to be"
<br /><br />
After training, as Andrej describes in the nanoGPT repo readme, the instruction says that you first run prepare.py, then "python train.py config/train_shakespeare_char.py", for this dataset (https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt
) . Files needed for sample.py are created -> meta.pkl and ckpt.pt. After the training phase, after the default 5000 iterations they are under these paths.
<br /><br />

```
nanoGPT-master\data\shakespeare_char\meta.pkl
nanoGPT-master\out-shakespear-char\ckpt.pt
```
And this is what the default sample.py script uses https://github.com/karpathy/nanoGPT/blob/master/sample.py
<br /><br />
<h3>Web Interface</h3>
I've only included 3 files here because you need app.py to run this. The demos below show the console version. You also need FLASK, etc.

```
pip install flask torch tiktoken
```

<br />
<b>Put app.py file ( and templates folder with index.html ) where sample.py, train.py is, i.e. MAIN folder.</b>
<br /><br />
Example of launching a web service for samples
<br /><br />

```
python app.py --out_dir=out-shakespear-char --device=cpu --port=5000
```

This is what you see in the terminal. Then open a browser at http://localhost:5000 

```
python app.py --out_dir=out-shakespear-char --device=cpu --port=5000
Loading model... (this may take a while)
number of parameters: 10.65M
Loading meta from data\shakespeare_char\meta.pkl...
Model loaded and ready.
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WS
GI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.101:5000
Press CTRL+C to quit
```

It looks just like the picture. Type the phrase in window on the left and press the generate button. And wait for a while. You will see the output on the right side like this.
<br />

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/images/110%20-%2012-03-2026%20-%20web%20service%20for%20nanoGPT.png?raw=true)

<br />
<b>DEMO 1 - sample_list_3.py script to run in console - example of action</b>

```
python sample_list_3.py --out_dir=out-shakespear-char --device=cpu
```

output - each line ends with <eot> generation

```
python sample_list_3.py --out_dir=out-shakespear-char --device=cpu
Overriding: out_dir = out-shakespear-char
Overriding: device = cpu
number of parameters: 10.65M
Loading meta from data\shakespeare_char\meta.pkl...
Enter prompt (finish with an empty line). Submit an empty prompt to quit.
To be, or not to be              # << Press ENTER after this ( after string )
                                 # << Then press ENTER a second time to make a blank line. 
 so, ere I receive,
To see the<eot> spoken; England's hands
And l<eot> thy wife, I think, and thy lo<eot> blood of love.

BUCKINGHAM:
A<eot> a subtle that
Which will weig<eot> my father's company.

KING RI<eot> it renown'd, I am darkness.
T<eot> rough more of the king;
The g<eot> resolved;
And come frown'd by<eot> seen, I ought your house
Whic<eot> received,
The sons of this fo<eot> the king.

KING RICHARD II:
I<eot> good to kill it.

DUKE OF YOR<eot> too poison!
Stay you well, it<eot> so distinied,
Who did grieve <eot> provoked in a score.

DUKE OF<eot> the greatest apparent,
Unless<eot> so aspect as he is now.

DUKE<eot> offer'd; I sing, that I have:<eot> no so for me: therefore, bese

Interrupted by user. Exiting.
```

<b>DEMO 2 - sample_list_4.py script to run in console - example of action</b>

```
python sample_list_4.py --out_dir=out-shakespear-char --device=cpu
Overriding: out_dir = out-shakespear-char
Overriding: device = cpu
number of parameters: 10.65M
Loading meta from data\shakespeare_char\meta.pkl...
Prompt (single line, empty to quit): To be, or not to be
To be, or not to be so, ere I receive,
To see the<eot>To be, or not to be gleeded to look in arms,
To b<eot>To be, or not to be time to the fool,
Nor nor fro<eot>To be, or not to begin the brain: yet, he's not.
<eot>To be, or not to be
This earth, for the self-stin<eot>To be, or not to be jealous to our fith.

CLARENC<eot>To be, or not to be gone:
But if I shall be recei<eot>To be, or not to be their complots not.

KING RIC<eot>To be, or not to be so find to be much a friend,
<eot>To be, or not to be ourselves, thou didst not not<eot>To be, or not to be slain.

ROMEO:
No, sweet such<eot>To be, or not to be here.

CORIOLANUS:
Why?

VOLU<eot>To be, or not to be so disputed,
The help and thr<eot>To be, or not to beg the same man there;
Then if <eot>To be, or not to be in his life.

LEONTES:
I woul<eot>To be, or not to be issued.

HENRY BOLINGBROKE:
N<eot>To be, or not to beget me to be horted
That may s<eot>To be, or not to be sent at so.

POMPEY:
Ha, how <eot>To be, or not to be your chamber:
My lords, I hav<eot>To be, or not to be of my justice's soul,
As lean

Interrupted by user. Exiting.
```


