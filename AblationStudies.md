## Albation Studies Results
The training_log directory contains all the ablation studies results shown in the paper.
By running precise_iou.py, you'll see a summary of the results.

model.py contains all our models.

After downloading the Cityscapes dataset, you can run train.py to evaluate our best model on the val set. 
You should see 78.85 mIOU.

Our final model exp48_decoder26 is what we call RegSeg.

There are a few notes about the results:

1. The ablation studies use the reduced mIOU (mIOU^R) while the results when comparing against competitors use the original mIOU.

2. The reduced mIOU shown in the training_log txt files are incorrect because it excludes only the train class; use precise_iou.py instead.

3. When doing ablation studies, the images are stored in half resolution and resized back to full resolution when loading, unless specified otherwise. 
   We store and load the images at full resolution when comparing against DDRNet-23 and when we submit to the test server.
   
4. A single .txt training file can contain multiple runs of the same config. precise_iou.py handles this for us.

5. For each .txt training file, there is a line at the beginning that shows the exact config used to train the model.
   
Below, we match the descriptions in our paper to the names of our backbones (body_name) and our decoders (decoder_name) in model.py.

Row | body_name | dilation rates | field-of-view | mIOU^R
--- | --- | --- | --- | ---
1 | exp48 | (1,1)+(1,2)+4*(1,4)+7*(1,14) | 3807 | 75.85
2 | exp43 | (1,1)+(1,2)+(1,4)+(1,6)+(1,8)+(1,10)+7*(1,12) | 3743 | 75.75
3 | exp50 | (1,1)+(1,2)+(1,4)+(1,6)+(1,8)+(1,10)+7*(1,3,6,12) | 3743 | 75.69
4 | exp46 | (1,1)+(1,2)+(1,4)+(1,6)+(1,8)+8*(1,10) | 3295 | 75.58
5 | exp49 | (1,1)+(1,2)+6*(1,4)+5*(1,6,12,18) | 3807 | 75.54
6 | exp52 | (1,1)+(1,2)+(1,4)+10*(1,6) | 2207 | 75.53
7 | exp47 | (1,1)+(1,2)+(1,4)+(1,6)+(1,8)+(1,10)+(1,12)+6*(1,14) | 4127 | 75.45
8 | exp30 | 5*(1,4)+8*(1,10) | 3263 | 75.44
9 | exp51 | (1,1)+(1,2)+(1,4)+(1,6)+(1,8)+(1,10)+7*(1,4,8,12) | 3743 | 75.27
10 | regnety600mf | 8*(1,1)+3*(2,2) | 607 | 73.25


decoder_name | Decoder | mIOU^R
--- | --- | ---
decoder26 | Sec 3.4 decoder | 75.84
decoder14 | sum+3x3 conv | 75.75
decoder10 | concat+Y block| 75.70
decoder4 | concat+3x3 conv | 75.62
decoder12 | sum+1x1 conv | 74.93
lraspp | LRASPP | 74.85
SFNetDecoder | SFNetDecoder |74.80
BisenetDecoder | BiSeNetDecoder | 74.68
