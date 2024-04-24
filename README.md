# Image-Super-Resolution-Model

Machine learning project on an image denoising/super-resolution app, by Dimitri Christopoulos, Evan Cureton, and Arnauld Binder.
Colab: https://colab.research.google.com/drive/1twpmFSHA_nZ9FrxE3deNwjqXmU3-8Ji4?usp=sharing

**Format of project:**

- ./dataset:
    - dataset.py, has dataset class.
    - LowerResolution.py, used for image blurring (blurred image using gaussian blur with radius 2).
    - utils.py, a helper function used to obtain image paths from desktop.

- ./model:
    - super_resolution.py which is the autoencoder model

- ./weights:
    - trained weights

- test.py
    - Python program used for testing

- train.py
    - Python program used for training
  
**Results:**

![Untitled presentation](https://github.com/DimitriC3/Image-Super-Resolution-Model/assets/154035020/1a465032-ff49-47ce-90bd-792417402c73)

