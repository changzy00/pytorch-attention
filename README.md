![image](https://github.com/changzy00/pytorch-attention/blob/master/images/logo.jpg)
# This codebase is a PyTorch implementation of various attention mechanisms, CNNs, Vision Transformers and MLP-Like models.

![](https://img.shields.io/badge/python->=v3.0-yellowgreen)

![](https://img.shields.io/badge/pytorch->=v1.5-yellowgreen)

If it is helpful for your work, please⭐

# Updating...

## Attention mechanisms

* Squeeze-and-Excitation Networks (CVPR 2018) [pdf](https://arxiv.org/pdf/1709.01507)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/senet.png)

* ##### Code
```python
import torch
from attention_mechanisms.se_module import SELayer

x = torch.randn(2, 64, 32, 32)
attn = SELayer(64)
y = attn(x)
print(y.shape)

```
* CBAM: convolutional block attention module (ECCV 2018) [pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cbam.png)

* ##### Code
```python
import torch
from attention_mechanisms.cbam import CBAM

x = torch.randn(2, 64, 32, 32)
attn = CBAM(64)
y = attn(x)
print(y.shape)
```
* Bam: Bottleneck attention module(BMVC 2018) [pdf](http://bmvc2018.org/contents/papers/0092.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/bam.png)

* ##### Code
```python
import torch
from attention_mechanisms.bam import BAM

x = torch.randn(2, 64, 32, 32)
attn = BAM(64)
y = attn(x)
print(y.shape)
```
* A2-nets: Double attention networks (NeurIPS 2018) [pdf](https://arxiv.org/pdf/1810.11579)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/a2net.png)

* ##### Code
```python
import torch
from attention_mechanisms.double_attention import DoubleAttention

x = torch.randn(2, 64, 32, 32)
attn = DoubleAttention(64, 32, 32)
y = attn(x)
print(y.shape)
```
* Srm : A style-based recalibration module for convolutional neural networks (ICCV 2019)  [pdf](https://arxiv.org/pdf/1903.10829)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/srm.png)

* ##### Code
```python
import torch
from attention_mechanisms.srm import SRM

x = torch.randn(2, 64, 32, 32)
attn = SRM(64)
y = attn(x)
print(y.shape)
```
* Gcnet: Non-local networks meet squeeze-excitation networks and beyond (ICCVW 2019) [pdf](https://arxiv.org/pdf/1904.11492)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gcnet.png)

* ##### Code
```python
import torch
from attention_mechanisms.gc_module import GCModule

x = torch.randn(2, 64, 32, 32)
attn = GCModule(64)
y = attn(x)
print(y.shape)
```
* Selective Kernel Networks (CVPR 2019) [pdf](https://arxiv.org/abs/1903.06586)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/sknet.png)

* ##### Code
```python
import torch
from attention_mechanisms.sk_module import SKLayer

x = torch.randn(2, 64, 32, 32)
attn = SKLayer(64)
y = attn(x)
print(y.shape)
```
* Linear Context Transform Block (AAAI 2020) [pdf](https://arxiv.org/pdf/1909.03834v2)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/lct.png)

* ##### Code
```python
import torch
from attention_mechanisms.lct import LCT

x = torch.randn(2, 64, 32, 32)
attn = LCT(64, groups=8)
y = attn(x)
print(y.shape)
```
* Gated Channel Transformation for Visual Recognition (CVPR 2020) [pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)
* Ecanet: Efficient channel attention for deep convolutional neural networks (CVPR 2020) [pdf](https://arxiv.org/pdf/1910.03151)
* Rotate to Attend: Convolutional Triplet Attention Module (WACV 2021) [pdf](http://arxiv.org/pdf/2010.03045)
* Gaussian Context Transformer (CVPR 2021) [pdf](http://openaccess.thecvf.com//content/CVPR2021/papers/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.pdf)
* Coordinate Attention for Efficient Mobile Network Design (CVPR 2021) [pdf](https://arxiv.org/abs/2103.02907)
* SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021) [pdf](http://proceedings.mlr.press/v139/yang21o/yang21o.pdf)

## Vision Transformers

* An image is worth 16x16 words: Transformers for image recognition at scale (ICLR 2021) [pdf](https://arxiv.org/pdf/2010.11929)
* XCiT: Cross-Covariance Image Transformer (NeurIPS 2021) [pdf](https://arxiv.org/pdf/2106.09681)
* Rethinking Spatial Dimensions of Vision Transformers (ICCV 2021) [pdf](https://arxiv.org/abs/2103.16302)
* CvT: Introducing Convolutions to Vision Transformers (ICCV 2021) [pdf](https://arxiv.org/abs/2103.15808)
* Pyramid vision transformer: A versatile backbone for dense prediction without convolutions (ICCV 2021) [pdf](https://arxiv.org/abs/2102.12122)
* CMT: Convolutional Neural Networks Meet Vision Transformers (CVPR 2022) [pdf](http://arxiv.org/pdf/2107.06263)
* MetaFormer is Actually What You Need for Vision (CVPR 2022) [pdf](https://arxiv.org/abs/2111.11418)
* KVT: k-NN Attention for Boosting Vision Transformers (ECCV 2022) [pdf](https://arxiv.org/abs/2106.00515)
* MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (ICLR 2022) [pdf](https://arxiv.org/abs/2110.02178)
* Pyramid Pooling Transformer for Scene Understanding (TPAMI 2022) [pdf](https://arxiv.org/abs/2106.12011)
* EfficientFormer: Vision Transformers at MobileNet Speed (NeurIPS 2022) [pdf](https://arxiv.org/abs/2212.08059)
* When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism (AAAI 2022) [pdf](https://arxiv.org/abs/2201.10801)
* CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows (CVPR 2022) [pdf](https://arxiv.org/pdf/2107.00652.pdf)
* DilateFormer: Multi-Scale Dilated Transformer for Visual Recognition (TMM 2023) [pdf](https://arxiv.org/abs/2302.01791)
* BViT: Broad Attention based Vision Transformer (TNNLS 2023) [pdf](https://arxiv.org/abs/2202.06268)
* MOAT: Alternating Mobile Convolution and Attention Brings Strong Vision Models (ICLR 2023) [pdf](https://arxiv.org/pdf/2210.01820.pdf)



## Convolutional Neural Networks(CNNs)

* Network In Network (ICLR 2014) [pdf](https://arxiv.org/pdf/1312.4400v3)
* Deep Residual Learning for Image Recognition (CVPR 2016) [pdf](https://arxiv.org/abs/1512.03385)
* Wide Residual Networks (BMVC 2016) [pdf](https://arxiv.org/pdf/1605.07146)
* Densely Connected Convolutional Networks (CVPR 2017) [pdf](http://arxiv.org/abs/1608.06993v5)
* Deep Pyramidal Residual Networks (CVPR 2017) [pdf](https://arxiv.org/pdf/1610.02915)
* MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (CVPR 2017) [pdf](https://arxiv.org/pdf/1704.04861.pdf)
* MobileNetV2: Inverted Residuals and Linear Bottlenecks (CVPR 2018) [pdf](https://arxiv.org/abs/1801.04381)
* Searching for MobileNetV3 (ICCV 2019) [pdf](https://arxiv.org/pdf/1905.02244)
* MnasNet: Platform-Aware Neural Architecture Search for Mobile (CVPR 2019) [pdf](http://arxiv.org/pdf/1807.11626)
* EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019) [pdf](https://arxiv.org/abs/1905.11946)
* Res2Net: A New Multi-scale Backbone Architecture (TPAMI 2019) [pdf](https://arxiv.org/pdf/1904.01169)
* Rethinking Bottleneck Structure for Efficient Mobile Network Design (ECCV 2020) [pdf](https://arxiv.org/pdf/2007.02269.pdf)
* GhostNet: More Features from Cheap Operations (CVPR 2020) [pdf](https://arxiv.org/abs/1911.11907)
* EfficientNetV2: Smaller Models and Faster Trainin (ICML 2021) [pdf](https://arxiv.org/abs/2104.00298)
* A ConvNet for the 2020s (CVPR 2022) [pdf](https://arxiv.org/abs/2201.03545)

## MLP-Like Models

* MLP-Mixer: An all-MLP Architecture for Vision (NeurIPS 2021) [pdf](https://arxiv.org/pdf/2105.01601.pdf)
* Pay Attention to MLPs (NeurIPS 2021) [pdf]( https://arxiv.org/pdf/2105.08050)
* Global Filter Networks for Image Classification (NeurIPS 2021) [pdf](https://arxiv.org/abs/2107.00645)
* Sparse MLP for Image Recognition: Is Self-Attention Really Necessary? (AAAI 2022) [pdf](https://arxiv.org/abs/2109.05422)
* DynaMixer: A Vision MLP Architecture with Dynamic Mixing (ICML 2022) [pdf](https://arxiv.org/pdf/2201.12083)
* Patches Are All You Need? (TMLR 2022) [pdf](https://arxiv.org/pdf/2201.09792)
* Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition (TPAMI 2022) [pdf](https://arxiv.org/abs/2106.12368)
* CycleMLP: A MLP-like Architecture for Dense Prediction (ICLR 2022) [pdf](https://arxiv.org/abs/2107.10224)
* Sequencer: Deep LSTM for Image Classification (NeurIPS 2022) [pdf](https://arxiv.org/abs/2205.01972)

