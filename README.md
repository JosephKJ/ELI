# Energy-based Latent Aligner for Incremental Learning

### Accepted to CVPR 2022 [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/)

<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/4231550/159659561-17bea6a6-5228-42e6-a811-eb18d37c48e9.png" width="500"/>
</p>
<p align="center" width="80%">
We illustrate an Incremental Learning model trained on a continuum of tasks in the top part of the figure. While learning the current task <img src="https://render.githubusercontent.com/render/math?math=\tau_t">, the latent representation of Task <img src="https://render.githubusercontent.com/render/math?math=\tau_{t-1}"> data gets disturbed, as shown by red arrows. ELI learns an energy manifold, and uses it to counteract this inherent representational shift, as illustrated by green arrows, thereby alleviating forgetting.</p>

### Overview

[//]: # (Deep learning models tend to forget their earlier knowledge while incrementally learning new tasks. This behavior emerges because the parameter updates optimized for the new tasks may not align well with the updates suitable for older tasks. The resulting latent representation mismatch causes forgetting. )

In this work, we propose ELI: Energy-based Latent Aligner for Incremental Learning, which:
- Learns an energy manifold for the latent representations such that previous task latents will have low energy and the current task latents have high energy values. 
- This learned manifold is used to counter the representational shift that happens during incremental learning.

The implicit regularization that is offered by our proposed methodology can be used as a **plug-and-play module** in existing incremental learning methodologies for classification and object-detection. 

[//]: # (We validate this through extensive evaluation on CIFAR-100, ImageNet subset, ImageNet 1k and Pascal VOC datasets. We observe consistent improvement when ELI is added to three prominent methodologies in class-incremental learning, across multiple incremental settings. )

[//]: # (Further, when added to the state-of-the-art incremental object detector, ELI provides over 5% improvement in detection accuracy, corroborating its effectiveness and complementary advantage to existing art.)


[//]: # (## Methodology)

[//]: # ()
[//]: # (<p align="center" width="100%">)

[//]: # (<img src="https://user-images.githubusercontent.com/4231550/159659616-23f6d790-35b3-4be3-b183-c5afda18e9d9.png" width="600"/>)

[//]: # (</p>)


## Toy Experiment

<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/4231550/159659669-be756c6b-1948-4cd1-9ab7-acec9c69030b.png"/>
</p>

A key hypothesis that we base our methodology is that while learning a new task, the latent representations will get disturbed, which will in-turn cause catastrophic forgetting of the previous task, and that an energy manifold can be used to align these latents, such that it alleviates forgetting. 

Here, we illustrate a proof-of-concept that our hypothesis is indeed true.
We consider a two task experiment on MNIST, where each task contains a subset of classes: <img src="https://render.githubusercontent.com/render/math?math=\tau_1"> = {0, 1, 2, 3, 4}, <img src="https://render.githubusercontent.com/render/math?math=\tau_2"> = {5, 6, 7, 8, 9}. 

After learning the second task, the accuracy on <img src="https://render.githubusercontent.com/render/math?math=\tau_1"> test set drops to 20.88%, while experimenting with a 32 dimensional latent space.
The latent aligner in ELI provides 62.56% improvement in test accuracy to 83.44%.
The visualization of a 512 dimensional latent space after learning <img src="https://render.githubusercontent.com/render/math?math=\tau_2"> in sub-figure (c), indeed shows cluttering due to representational shift. ELI is able to align the latents as shown in sub-figure (d), which alleviates the drop in accuracy from 89.14% to 99.04%.

The code for these toy experiments are in:
- [ELI.ipynb](https://github.com/JosephKJ/ELI/blob/main/ELI.ipynb)
- [ELI_512.ipynb](https://github.com/JosephKJ/ELI/blob/main/ELI_512.ipynb)


## Implicitly Recognizing and Aligning Important Latents

https://user-images.githubusercontent.com/4231550/159675403-f2cee8e3-bddb-4e8f-80a1-90cb638b372e.mp4

Each row <img src="https://render.githubusercontent.com/render/math?math=i"> shows how <img src="https://render.githubusercontent.com/render/math?math=i^th"> latent dimension is updated by ELI. We see that different dimensions have different degrees of change, which is implicitly decided by our energy-based model.


## Classification and Detection Experiments

Code and models for the classification and object detection experiments are inside the respective folders:

- [classification](https://github.com/JosephKJ/ELI/tree/main/classification)
- [detection](https://github.com/JosephKJ/ELI/tree/main/detection)

Each of these are independent repositories. Please consider them separate. 

## Citation
If you find our research useful, please consider citing us:

```BibTeX

@inproceedings{joseph2022Energy,
  title={Energy-based Latent Aligner for Incremental Learning},
  author={Joseph, KJ and Khan, Salman and Khan, Fahad Shahbaz and Anwar, Rao Muhammad and Balasubramanian, Vineeth},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```


## Our Related Work
- Open-world Detection Transformer, CVPR 2022. [Paper](https://arxiv.org/pdf/2112.01513.pdf) | [Code]()
- Towards Open World Object Detection, CVPR 2021. (Oral) [Paper](https://arxiv.org/abs/2103.02603) | [Code](https://github.com/JosephKJ/OWOD)
- Incremental Object Detection via Meta-learning, TPAMI 2021. [Paper](https://arxiv.org/abs/2003.08798) | [Code](https://github.com/JosephKJ/iOD)
