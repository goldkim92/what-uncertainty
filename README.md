# What uncertainties
Pytorch implementation of ["What uncertainties do we need in bayesian deep learning for computer vision"](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) 
<br>
This code is implemented only for the image classification task (Cifar100).

## Prerequisites
* Python 3.6
* Pytorch 0.4.0

## Train
To train the model with Cifar100 dataset, run the code by
```
$ nohup python main.py --gpu_number=0 --arch=resnet_aleo & 
```
Log is updated and the checkpoint is saved in `./runs/resnet_aleo/` directory. <br>
To train the resnet baseline model, change the `--arch` argument to `resnet`.

## Results
We show the experiments in this section. 
We first compare the proposed network (valid acc. 71.1%) with the baseline resnet network (valid acc. 70.7%) by observing the calibration effect. 
After that, we compare the aleotoric uncertainty and the epistemic uncertainty in the out-of-distribution (Cifar100 vs SVHN) setting.
### Calibration

<p align="center">
  <img src="./png/id_comparison.png">
</p>
<!-- ![](png/id_comparison.png) -->

We first compare the proposed network and the baseline network by the calibration benchmarks. 
The first figure at the above shows that the propose network has lower expected calibration error (ECE) (lower is better), though it is not significantly different. 
The second and third figures provide the comparison in the proper scoring rules (NLL and Brier score) (lower is better) when input images are rotated. 
In the in-distribution setting (when rotation angle is 0), the propsed network has better performance than the baseline network, where the proposed network has 1.47 and 0.44 for NLL and Brier score, respectively, and the baseline network has 1.50 and 0.45 for NLL and Brier score, respectively. 
However, when the input images are regarded to be sampled from out-of-distribution by rotating the original images, the baseline network has better performance in the calibration. 
To sum up, The proposed method has a little effect on calibration.

### Out-of-distribution
<p align="center">
  <img src="./png/ood_comparison.png">
</p>
