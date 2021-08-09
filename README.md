# Bi_S_LSTM
Partial Implementation of Image Captioning with Deep Bidirectional LSTMs


#### Note: this is a work in progress. I will upload results with other datasets and a detailed explanation soon.

### Introduction:
This model is similar to Bi-S-LSTM model proposed in _Image Captioning with Deep Bidirectional LSTMs_ published in _24th ACM international conference on Multimedia_ [[link]](https://dl.acm.org/doi/abs/10.1145/2964284.2964299) . An updated paper was published in journal ACM TOMM [[link]](https://dl.acm.org/doi/abs/10.1145/3115432).
There are following differences in our implementation:
1. I have not used Data Augmentation in this implementation. However, I have included options for horizontal and vertical data augmentation in the code which can be used by setting use_data_augmentation = True in train.py.
2. I have used batch size of 32 for all experiments and learning rate of 0.0001.
3. I have used VGG-16 CNN for image feature extraction whereas the authors used both AlexNet and VGG-16 for experiments.
4. Since both forward and backward LSTMs are trained for caption generation, I have experimented with both the inference strategy used in the paper (where the most likely sentence generated by forward or backward LSTMs is used as caption) and separate inference with backward and forward LSTMs.
5. I could not find hidden and cell state initialization in the paper. So, I have initilized hidden and cell states as zero vectors for both forward and backward LSTMs.

### Method
I have used three-layered stacked Bi-Directional LSTM framework as described in paper. The first two layers are Text-LSTM (T-LSTM) and the third layer is Multimodal-LSTM (M-LSTM). The first T-LSTM layer takes as input, the word vector representations. The second T-LSTM layer receives hidden state of previous layer as input. 
The output of second Text-LSTM (T-LSTM) layer and Image Feature representation are concatenated together before being used as input to Multimodal-LSTM (M-LSTM). 
It is mentioned in the paper that M-LSTM uses both image representation and T-LSTM hidden state but it's not clear to me how both quantities are used. So I have merged them by concatenation and fed them as input to M-LSTM. I have evaluated with 1, 3, 5, 10, 15 and 20 as beam sizes. In the paper, authors have used greedy evaluation with beam size as 1.

In the paper, both forward and backward LSTMs are trained to generate captions and their losses are combined. During evaluation the captions generated by forward and backward LSTM are evaluated and most likely caption is selected at each time-step. In our implementation, we save captions generated by both forward and backward LSTMs separately and also record caption generated by overall model (i.e., the most likely caption, from backward and forward LSTMs, recorded at each time-step.) 

### Results

**For Flickr8k dataset:**

The following table contains results obtained from overall model (best captions selected from forward and backward LSTMs):

|Result |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
|Paper | 1 | 0.642 | 0.443 | 0.292 | 0.186 |  |  |  |  |
|Our | 1 | 0.624 | 0.415 | 0.262 | 0.162 | 0.189 | 0.380 | 0.120 | 0.434 |
|Our | 3 | 0.583 | 0.389 | 0.244 | 0.146 | 0.161 | 0.341 | 0.110 | 0.410 |
|Our | 5 | 0.571 | 0.381 | 0.238 | 0.141 | 0.156 | 0.336 | 0.105 | 0.402 |
|Our | 10 | 0.549 | 0.363 | 0.226 | 0.135 | 0.150 | 0.317 | 0.103 | 0.397 |
|Our | 15 | 0.529 | 0.351 | 0.216 | 0.125 | 0.146 | 0.305 | 0.102 | 0.390 |
|Our | 20 | 0.520 | 0.346 | 0.216 | 0.127 | 0.146 | 0.310 | 0.103 | 0.389 |

Following table contains results obtained using captions generated by forward LSTM only:

|Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.596 | 0.394 | 0.243 | 0.150 | 0.178 | 0.350 | 0.112 | 0.416 |
| 3 | 0.601 | 0.403 | 0.258 | 0.165 | 0.179 | 0.376 | 0.120 | 0.422 |
| 5 | 0.604 | 0.407 | 0.259 | 0.164 | 0.175 | 0.376 | 0.118 | 0.423 |
| 10 | 0.611 | 0.413 | 0.265 | 0.170 | 0.173 | 0.385 | 0.119 | 0.425 |
| 15 | 0.613 | 0.412 | 0.263 | 0.167 | 0.172 | 0.375 | 0.118 | 0.423 |
| 20 | 0.615 | 0.413 | 0.262 | 0.164 | 0.171 | 0.372 | 0.117 | 0.423 |


Following table contains results obtained using captions generated by backward LSTM only:

|Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.600 | 0.389 | 0.242 | 0.147 | 0.193 | 0.361 | 0.120 | 0.428 |
| 3 | 0.576 | 0.386 | 0.242 | 0.142 | 0.161 | 0.337 | 0.112 | 0.408 |
| 5 | 0.565 | 0.375 | 0.232 | 0.134 | 0.154 | 0.326 | 0.106 | 0.398 |
| 10 | 0.544 | 0.358 | 0.221 | 0.127 | 0.149 | 0.310 | 0.103 | 0.395 |
| 15 | 0.525 | 0.347 | 0.212 | 0.120 | 0.145 | 0.301 | 0.101 | 0.388 |
| 20 | 0.514 | 0.342 | 0.211 | 0.121 | 0.144 | 0.301 | 0.101 | 0.386 |


### Reproducing the results:
1. Download 'Karpathy Splits' for train, validation and testing from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
2. For evaluation, the model already generates BLEU scores. In addition, it saves results and image annotations as needed in MSCOCO evaluation format. So for generation of METEOR, CIDEr, ROUGE-L and SPICE evaluation metrics, the evaluation code can be downloaded from [here](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI).

#### Prerequisites:
1. This code has been tested on python 3.6.9 but should word on all python versions > 3.6.
2. Pytorch v1.5.0
3. CUDA v10.1
4. Torchvision v0.6.0
5. Numpy v.1.15.0
6. pretrainedmodels v0.7.4 (Install from [source](https://github.com/Cadene/pretrained-models.pytorch.git)). (I think all versions will work but I have listed here for the sake of completeness.)


#### Execution:
1. First set the path to Flickr8k/Flickr30k/MSCOCO data folders in create_input_files_dataname.py file ('dataname' replaced by f8k/f30k/coco).
2. Create processed dataset by running: 
> python create_input_files_dataname.py

3. To train the model:
> python train_dataname.py

4. To evaluate: 
> python eval_dataname.py beamsize 

(eg.: python train_f8k.py 20)
