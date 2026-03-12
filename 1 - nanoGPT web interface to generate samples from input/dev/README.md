<h3>TODO - several metrics are needed to measure errors in prediction sequence  (length) etc.</h3>
The default script that trains a model of 10.67M parameters has a context of 256 tokens https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py#L19
<br /><br />
Tiny Shakespeare is ok, but what if you change the input.txt which is created after running prepare.py here https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char
<br /><br />
What if I change the content of input.txt to another text e.g. https://uefi.org/sites/default/files/resources/UEFI_Spec_2_10_Aug29.pdf. 
<br /><br />

Tiny Shakespeare 
```
length of dataset in characters: 1,115,394
all the unique characters: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab size: 65
train has 1,003,854 tokens
val has 111,540 tokens
```

Just CTRL + A and CTRL + V to notepad from UEFI_Spec_2_10_Aug29.pdf to create dataset and replace input.txt 

```
length of dataset in characters: 4,806,534
all the unique characters: \u0002\
!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¥§©®ÂÄÊÎÔÛàáâäêîðôû˓–—‘’“”„•™→
vocab size: 127
train has 4,325,880 tokens
val has 480,654 tokens
```

~ 4x times more looking quickly 
<br /><br />
For example <b>EFI IPv4 Protocol</b>, I want 25 predictions (5 lines x 5 samples ) for the phrase "EFI IPv4 Protocol". Next tokens : 30

```
Generated output
Line 1:
EFI IPv4 Protocol 1151
Unified Extensible Firmw<eot>EFI IPv4 Protocol driver instance.
28.8. EFI IP<eot>EFI IPv4 Protocol instance are not available.
E<eot>EFI IPv4 Protocol instance and configures a pro<eot>EFI IPv4 Protocol instance.
Udp4ConfigData.Stat

Line 2:
EFI IPv4 Protocol 1137
Unified Extensible Firmw<eot>EFI IPv4 Protocol instance.
Refer to the type o<eot>EFI IPv4 Protocol 1110
Unified Extensible Firmw<eot>EFI IPv4 Protocol 1150
Unified Extensible Firmw<eot>EFI IPv4 Protocol instance. If the protocol ins

Line 3:
EFI IPv4 Protocol driver instance is returned i<eot>EFI IPv4 Protocol instance must be configured d<eot>EFI IPv4 Protocol instance.
OptionsLength Lengt<eot>EFI IPv4 Protocol driver will check successfull<eot>EFI IPv4 Protocol instance has not been initial

Line 4:
EFI IPv4 Protocol instance.
ServiceType Set the<eot>EFI IPv4 Protocol 1128
Unified Extensible Firmw<eot>EFI IPv4 Protocol 1352
Unified Extensible Firmw<eot>EFI IPv4 Protocol .
Description
The EFI_IP4_PRO<eot>EFI IPv4 Protocol instance.
Description
The EFI

Line 5:
EFI IPv4 Protocol Child instance. When the user<eot>EFI IPv4 Protocol driver. The driver will then <eot>EFI IPv4 Protocol instance.
Token Pointer to th<eot>EFI IPv4 Protocol instance will be sent to the <eot>EFI IPv4 Protocol instance.
NotifyConfiguration
```

"EFI IPv4 Protocol driver instance" etc can be found in the document, so even though the model predicts char by char, it can recognize 2 words. Maybe more? Not bad.
<h3>Summary</h3>
But this is for more tests and build some metrics to measure this kind of things.
