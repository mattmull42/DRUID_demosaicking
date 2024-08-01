# Pattern-invariant Unrolling for Robust Demosaicking

This is the code repository for the paper _Pattern-invariant Unrolling for Robust Demosaicking_ submitted at the EUSIPCO - 2024 conference.

## Overview

To acquire color images, most commercial cameras
rely on color filter arrays (CFAs), which are a pattern of
color filters overlaid over the sensor’s focal plane. Demosaicking
describes the processing techniques aimed at the reconstruction
of a full color image for all pixels on the focal plane array.
Most demosaicking methods are tailored for a specific CFA and
therefore tend to work poorly for other ones. This work
presents an algorithm for demosaicking a wide variety of CFA.
The proposed method allows to blend the knowledge of the
CFA with information coming from data, employing a novel
transformation and pattern-invariant loss function. The method
is based on the unrolling of an algorithm based on a neural
network learned on available examples. Preliminary experiments
over RGB and RGBW CFAs show that the method performs
well over a range of CFAs and is competitive for CFAs that the
competing methods were tailored to work well on.

## Installation

All the code run in Python 3.10+ and rely on classical Python libraries (torch, matplotlib, ...). The code is proposed in a Pytorch Lightning class for convenience in training, testing and predicting. However, it is easily possible to get only the network which is written in pure Pytorch. It is recommended to use a **virtual environment** to run the code.

```
git clone --recurse-submodules https://github.com/mattmull42/unrolled_demosaicking
cd unrolled_demosaicking
pip install -r requirements.txt
```

We provide in `weights` the trained parameters with and without invariance. In both cases the training was done on the same dataset (BSD500) with the same set of CFAs (Bayer, Chakrabarti, Gindele, Hamilton, Honda, Kaizu, Kodak, Quad-Bayer, Sony, Sparse3, Wang, Yamagami, Yamanaka) and with the same parameters.

To reproduce the results and use the proposed method in the paper the users need to use the weights ending by 'V' (which stands for Variations). Those weights come from the training with the invariance procedure enabled.

## Training

The Python script `train.py` contains all the routine to train the network on a chosen set of CFAs. It is highly recommended running additional trainings on a GPU. The best weights will be saved in the `weights` directory, in function of the names of the CFAs seen during the training. The letter 'V' is appended if the training is done with invariance enabled.

## Testing

The notebook `test.ipynb` provides the tools to apply the network with the selected weights on a dataset with a given set of CFAs. It gives the mean PSNR (dB) and SSIM, along with their corresponding standard variations.

## Predicting

The notebook `predict.ipynb` allows the users to apply the wanted weights to a specific image and retrieve its corresponding output with some utilities to present the results. The method is run on the chosen CFAs.

## Contact

For any questions please open an issue on this repository or contact us directly through our email addresses:

- matthieu.muller@gipsa-lab.fr
- daniele.picone@gipsa-lab.fr
- mauro.dalla-mura@gipsa-lab.fr
- mou@hi.is
