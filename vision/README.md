# Sematic Segmentation for Fence Detection

This folder explores the use of sematic segmentation methods for segmenting the fence in a image.

The dataset [Fence Segmentation Dataset](https://github.com/chen-du/De-fencing) is used to train the different models. Examples of the used data can be seen below:

| Real image | Real mask |
|:-:|:-:|
| ![](segnet/assets/data/real_image.png) | ![](segnet/assets/data/real_mask.png) |

## U-Net

A U-Net model is trained on both the [Fence Segmentation Dataset](https://github.com/chen-du/De-fencing).

### Results

## SegNet

A SegNet model is trained on both the [Fence Segmentation Dataset](https://github.com/chen-du/De-fencing).

### Results

## References

- [Deep learning based fence segmentation andremoval from an image using a video sequence](https://arxiv.org/pdf/1609.07727.pdf)
- [My camera can see through fences: A deep learningapproach for image de-fencing](https://arxiv.org/pdf/1805.07442.pdf)
- [Single-Image Fence Removal Using DeepConvolutional Neural Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8933392)
- [U-Net: Convolutional Networks for BiomedicalImage Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [SegNet: A Deep Convolutional Encoder-DecoderArchitecture for Image Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)
- [Fence Segmentation Dataset](https://github.com/chen-du/De-fencing)
- [synth-ml](https://gitlab.com/sdurobotics/vision/synth-ml)
