![image](https://github.com/changzy00/pytorch-attention/blob/master/images/logo.jpg)
# This codebase is a PyTorch implementation of various attention mechanisms, CNNs, Vision Transformers and MLP-Like models.

![](https://img.shields.io/badge/python->=v3.0-yellowgreen)

![](https://img.shields.io/badge/pytorch->=v1.5-yellowgreen)

If it is helpful for your work, please⭐

# Updating...

# Content

- [Attention mechanisms](#Attention mechanisms)
    - [1. Squeeze-and-Excitation Attention](#1-squeeze-and-excitation-attention)

    - [Convolutional Block Attention](#2-convolutional-block-attention)

## Attention mechanisms
### 1. Squeeze-and-Excitation Attention
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
* Convolutional Block Attention
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
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gate_channel.png)

* ##### Code
```python
import torch
from attention_mechanisms.gate_channel_module import GCT

x = torch.randn(2, 64, 32, 32)
attn = GCT(64)
y = attn(x)
print(y.shape)
```
* Ecanet: Efficient channel attention for deep convolutional neural networks (CVPR 2020) [pdf](https://arxiv.org/pdf/1910.03151)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/ecanet.png)

* ##### Code
```python
import torch
from attention_mechanisms.eca import ECALayer

x = torch.randn(2, 64, 32, 32)
attn = ECALayer(64)
y = attn(x)
print(y.shape)
```
* Rotate to Attend: Convolutional Triplet Attention Module (WACV 2021) [pdf](http://arxiv.org/pdf/2010.03045)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/triplet.png)

* ##### Code
```python
import torch
from attention_mechanisms.triplet_attention import TripletAttention

x = torch.randn(2, 64, 32, 32)
attn = TripletAttention(64)
y = attn(x)
print(y.shape)
```
* Gaussian Context Transformer (CVPR 2021) [pdf](http://openaccess.thecvf.com//content/CVPR2021/papers/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/gct.png)

* ##### Code
```python
import torch
from attention_mechanisms.gct import GCT

x = torch.randn(2, 64, 32, 32)
attn = GCT(64)
y = attn(x)
print(y.shape)
```
* Coordinate Attention for Efficient Mobile Network Design (CVPR 2021) [pdf](https://arxiv.org/abs/2103.02907)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/coordinate.png)

* ##### Code
```python
import torch
from attention_mechanisms.coordatten import CoordinateAttention

x = torch.randn(2, 64, 32, 32)
attn = CoordinateAttention(64, 64)
y = attn(x)
print(y.shape)
```
* SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021) [pdf](http://proceedings.mlr.press/v139/yang21o/yang21o.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/simam.png)

* ##### Code
```python
import torch
from attention_mechanisms.simam import simam_module

x = torch.randn(2, 64, 32, 32)
attn = simam_module(64)
y = attn(x)
print(y.shape)
```
## Vision Transformers

* An image is worth 16x16 words: Transformers for image recognition at scale (ICLR 2021) [pdf](https://arxiv.org/pdf/2010.11929)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/vit.png)

* ##### Code
```python
import torch
from vision_transformers.ViT import VisionTransformer

x = torch.randn(2, 3, 224, 224)
model = VisionTransformer()
y = model(x)
print(y.shape) #[2, 1000]
```
* XCiT: Cross-Covariance Image Transformer (NeurIPS 2021) [pdf](https://arxiv.org/pdf/2106.09681)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/xcit.png)

* ##### Code
```python
import torch
from vision_transformers.xcit import xcit_nano_12_p16
x = torch.randn(2, 3, 224, 224)
model = xcit_nano_12_p16()
y = model(x)
print(y.shape)
```
* Rethinking Spatial Dimensions of Vision Transformers (ICCV 2021) [pdf](https://arxiv.org/abs/2103.16302)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/pit.png)

* ##### Code
```python
import torch
from vision_transformers.pit import pit_ti
x = torch.randn(2, 3, 224, 224)
model = pit_ti()
y = model(x)
print(y.shape)
```
* CvT: Introducing Convolutions to Vision Transformers (ICCV 2021) [pdf](https://arxiv.org/abs/2103.15808)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cvt.png)

* ##### Code
```python
import torch
from vision_transformers.cvt import cvt_13
x = torch.randn(2, 3, 224, 224)
model = cvt_13()
y = model(x)
print(y.shape)
```
* Pyramid vision transformer: A versatile backbone for dense prediction without convolutions (ICCV 2021) [pdf](https://arxiv.org/abs/2102.12122)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/pvt.png)

* ##### Code
```python
import torch
from vision_transformers.pvt import pvt_t
x = torch.randn(2, 3, 224, 224)
model = pvt_t()
y = model(x)
print(y.shape)
```
* CMT: Convolutional Neural Networks Meet Vision Transformers (CVPR 2022) [pdf](http://arxiv.org/pdf/2107.06263)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cmt.png)

* ##### Code
```python
import torch
from vision_transformers.cmt import cmt_ti
x = torch.randn(2, 3, 224, 224)
model = cmt_ti()
y = model(x)
print(y.shape)
```
* MetaFormer is Actually What You Need for Vision (CVPR 2022) [pdf](https://arxiv.org/abs/2111.11418)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/poolformer.png)

* ##### Code
```python
import torch
from vision_transformers.poolformer import poolformer_12
x = torch.randn(2, 3, 224, 224)
model = poolformer_12()
y = model(x)
print(y.shape)
```
* KVT: k-NN Attention for Boosting Vision Transformers (ECCV 2022) [pdf](https://arxiv.org/abs/2106.00515)
* ##### Code
```python
import torch
from vision_transformers.kvt import KVT
x = torch.randn(2, 3, 224, 224)
model = KVT()
y = model(x)
print(y.shape)
```
* MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (ICLR 2022) [pdf](https://arxiv.org/abs/2110.02178)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/mobilevit.png)

* ##### Code
```python
import torch
from vision_transformers.mobilevit import mobilevit_s
x = torch.randn(2, 3, 224, 224)
model = mobilevit_s()
y = model(x)
print(y.shape)
```
* Pyramid Pooling Transformer for Scene Understanding (TPAMI 2022) [pdf](https://arxiv.org/abs/2106.12011)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/p2t.png)

* ##### Code
```python
import torch
from vision_transformers.p2t import p2t_tiny
x = torch.randn(2, 3, 224, 224)
model = p2t_tiny()
y = model(x)
print(y.shape)
```
* EfficientFormer: Vision Transformers at MobileNet Speed (NeurIPS 2022) [pdf](https://arxiv.org/abs/2212.08059)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/efficientformer.png)

* ##### Code
```python
import torch
from vision_transformers.efficientformer import efficientformer_l1
x = torch.randn(2, 3, 224, 224)
model = efficientformer_l1()
y = model(x)
print(y.shape)
```
* When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism (AAAI 2022) [pdf](https://arxiv.org/abs/2201.10801)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/shiftvit.png)

* ##### Code
```python
import torch
from vision_transformers.shiftvit import shift_t
x = torch.randn(2, 3, 224, 224)
model = shift_t()
y = model(x)
print(y.shape)
```
* CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows (CVPR 2022) [pdf](https://arxiv.org/pdf/2107.00652.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/cswin.png)

* ##### Code
```python
import torch
from vision_transformers.cswin import CSWin_64_12211_tiny_224
x = torch.randn(2, 3, 224, 224)
model = CSWin_64_12211_tiny_224()
y = model(x)
print(y.shape)
```
* DilateFormer: Multi-Scale Dilated Transformer for Visual Recognition (TMM 2023) [pdf](https://arxiv.org/abs/2302.01791)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/dilateformer.png)

* ##### Code
```python
import torch
from vision_transformers.dilateformer import dilateformer_tiny
x = torch.randn(2, 3, 224, 224)
model = dilateformer_tiny()
y = model(x)
print(y.shape)
```
* BViT: Broad Attention based Vision Transformer (TNNLS 2023) [pdf](https://arxiv.org/abs/2202.06268)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/bvit.png)

* ##### Code
```python
import torch
from vision_transformers.bvit import BViT_S
x = torch.randn(2, 3, 224, 224)
model = BViT_S()
y = model(x)
print(y.shape)
```
* MOAT: Alternating Mobile Convolution and Attention Brings Strong Vision Models (ICLR 2023) [pdf](https://arxiv.org/pdf/2210.01820.pdf)
* ##### Model Overview
![](https://github.com/changzy00/pytorch-attention/blob/master/images/moat.png)

* ##### Code
```python
import torch
from vision_transformers.moat import moat_0
x = torch.randn(2, 3, 224, 224)
model = moat_0()
y = model(x)
print(y.shape)
```


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

