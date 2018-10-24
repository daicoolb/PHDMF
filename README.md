[![](https://jaywcjlove.github.io/sb/ico/awesome.svg)](#) [![](https://jaywcjlove.github.io/sb/license/mit.svg)](#)

## PHDMF

> Collaborative Filtering(CF), a well-known approach in producing recommender systems,
has achieved wide use and excellent performance not only in research but also in industry.
However, problems related to cold start and data sparsity have caused CF to attract an
increasing amount of attention in efforts to solve these problems. Traditional approaches
adopt side information to extract effective latent factors but still have some room for
growth. Due to the strong characteristic of feature extraction in deep learning, many
researchers have employed it with CF to extract effective representations and to enhance its
performance in rating prediction. Based on this previous work, we propose a probabilistic
model that combines a stacked denoising autoencoder and a convolutional neural network
together with auxiliary side information (i.e, both from users and items) to extract users and
items' latent factors, respectively. Extensive experiments for four datasets demonstrate that
our proposed model outperforms other traditional approaches and deep learning models
making it state of the art.

### Updates for the data format
- Many people ask about the data format about the model. Our model has two kinds of data, one is `user side information` and the other is `item side information`.
- For `user side information`, the dataformat is `user_id::binary_value`, eg. `456::0010010000100000`
- For `item side information`, the dataformat is `user_id::item_id::rating`, eg. `456::1::3`
- Just see `run_test_preprocess.sh` for the process of data. Because the original data is too large, I cannot give a dowonload link.

### Updates for the use of the model
- 1. run `run_test_preprocess.sh` for process the data
- 2. run `run_test_PHDMF.sh` for training the data
- 3. run `python rmse.py` for test the performance of the model


### Note
If you find this model is useful for your research, please cite this [paper](http://proceedings.mlr.press/v77/liu17a/liu17a.pdf):
- PHD: A Probabilistic Model of Hybrid Deep Collaborative Filtering for Recommender Systems ACML 2017

### 1 PHD Model
![PHD Model](https://github.com/daicoolb/PHDMF/blob/master/PHD.png)

### 2 CNN Model
![CNN model](https://github.com/daicoolb/PHDMF/blob/master/CNN.png)

### 3 aSDAE Model
![aSDAE model](https://github.com/daicoolb/PHDMF/blob/master/aSDAE.png)

This is a variant of ConvMF and aSDAE. Certainly, it is based on [ConvMF](http://dm.postech.ac.kr/~cartopy/ConvMF/) and [aSDAE](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14676/13916).

We use aSDAE and CNN to generate the user latent factor and item latent factor, respectively.

If you want to use it, pleae install [keras](keras.io) and [tensorflow](http://tensorflow.org/) ,respectively.

Note that this model can deal with three conditions: 
- only user side information (aSDAE model)
- only item side information (ConvMF model)
- user and item side information (PHD model)

Tips：Please make sure you have a good deep learning environment to run these codes.
