// 17-03-2026

<h3>Few predictions from default nanoGPT for Tiny Shakespeare model but using EKD2 source code as input.txt</h3>
In an interviews with one of the creators of "Attention is all you need" when outlined history of LLM and transformer development. What led to this, and 
what tests and benchmarks were used. In one such interview, I heard that initially, benchmark was a translator. But then they realized they could train a large 
model with more parameters, for example 175B (GPT-3), which itself contains information about translations, e.g., FR <> EN. And then simply insert the first lines into the 
  context the model generated sensible answers. The conclusion was that it's worth training a general language model (transformer). To be more precise, I'm referring to this 
  interview https://www.youtube.com/watch?v=U1dozb0xQGc [ ML in PL - Lukasz Kaiser – Transformers - How Far Can They Go? ] 
<br /><br />
This is why after training as a base model on OpenWebText after a finetune phase ( https://github.com/karpathy/nanoGPT?tab=readme-ov-file#finetuning ) for tiny shakespeare 
   model behaves differently, i.e. it generates a better, more meaningful text. So Andrej showed method - how to do it. If I understand correctly now.
<br /><br />
But more computing is needed, more expensive hardware... my goals are different now.

<h3>Demo</h3>

