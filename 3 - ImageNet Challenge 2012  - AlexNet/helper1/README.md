<h3>This isn't exactly what AlexNet does on the first layer. It's just an idea for building helpers in HTML and JavaScript to visualize calculations.</h3>
<b>These 2 helpers are far from what the first layer of AlexNet does exactly, but to start somewhere.</b>
<br /><br />
This is to answer the questions: What does the MaxPooling layer do and why is it there, or what results does it produce?
<br /><br />
But as I wrote, it's not exactly what's happening on this layer, but to have some code that tries to visualize (animate) the results of these operations in a simple way. TODO
<br /><br />
demo2.html - a simple demo that generates random values ​​on the left and calculates patches (kernels) for output from the left grid to the right grid with stride 4 11x11 to 1 Relu output. <br /> <br />
demo3 - conv2d.html - a demo that patches 11x11 to 3x3 <br /> <br />
script.py - to manually check if there are actually 55 kernels on the next layer <br /> <br />

The image shows demo2.html. An example of visualizing 11x11 patches that produces 1 output for each patch with 4 stride sizes. To get a rough idea, how this compute grids of pixels in next layers using some kernels, filters, and different layer to compute something like pooling etc.

![dump](https://github.com/KarolDuracz/Machine-Learning/blob/main/images/alexnet%20helper1.png?raw=true)

 // 21-04-2026 - Maybe this is wrong approach to visualization. But for now, just to have something preliminary in this repo. <br />
 But considering what AlexNet does and the overall approach, the network training algorithm is crucial. 
