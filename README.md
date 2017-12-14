# PHDMF

![PHD Model](https://github.com/daicoolb/PHDMF/blob/master/PHD.png)

![CNN model](https://github.com/daicoolb/PHDMF/blob/master/CNN.png)

![aSDAE model](https://github.com/daicoolb/PHDMF/blob/master/aSDAE.png)

This is a variant of ConvMF and aSDAE. Certainly, it is based on [ConvMF](http://dm.postech.ac.kr/~cartopy/ConvMF/) and [aSDAE](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14676/13916).

We use aSDAE and CNN to generate the user latent factor and item latent factor, respectively.

If you want to use it, pleae install [keras](keras.io) and [tensorflow](http://tensorflow.org/) ,respectively.

Note that this model can deal with three conditions: 
- only user side information (aSDAE model)
- only item side information (ConvMF model)
- user and item side information (PHD model)

Tips：Please make sure you have a good deep learning environment to run these codes.
