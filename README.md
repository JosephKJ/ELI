# Energy-based Latent Aligner for Incremental Learning

### Accepted to CVPR 2022

<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/4231550/159659561-17bea6a6-5228-42e6-a811-eb18d37c48e9.png" width="500"/>
</p>
<p align="center" width="80%">
We illustrate an Incremental Learning model trained on a continuum of tasks in the top part of the figure. While learning the current task T_t, the latent representation of Task T_{t-1} data gets disturbed, as shown by red arrows. ELI learns an energy manifold, and uses it to counteract this inherent representational shift, as illustrated by green arrows, thereby alleviating forgetting.</p>

#### Abstract
Deep learning models tend to forget their earlier knowledge while incrementally learning new tasks. This behavior emerges because the parameter updates optimized for the new tasks may not align well with the updates suitable for older tasks. The resulting latent representation mismatch causes forgetting. 

In this work, we propose ELI: Energy-based Latent Aligner for Incremental Learning, which:
- Learns an energy manifold for the latent representations such that previous task latents will have low energy and the current task latents have high energy values. 
- This learned manifold is used to counter the representational shift that happens during incremental learning.

The implicit regularization that is offered by our proposed methodology can be used as a **plug-and-play module** in existing incremental learning methodologies. 

We validate this through extensive evaluation on CIFAR-100, ImageNet subset, ImageNet 1k and Pascal VOC datasets. We observe consistent improvement when ELI is added to three prominent methodologies in class-incremental learning, across multiple incremental settings. 

Further, when added to the state-of-the-art incremental object detector, ELI provides over 5% improvement in detection accuracy, corroborating its effectiveness and complementary advantage to existing art.


[//]: # (## Methodology)

[//]: # ()
[//]: # (<p align="center" width="100%">)

[//]: # (<img src="https://user-images.githubusercontent.com/4231550/159659616-23f6d790-35b3-4be3-b183-c5afda18e9d9.png" width="600"/>)

[//]: # (</p>)


## Toy Experiment

<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/4231550/159659669-be756c6b-1948-4cd1-9ab7-acec9c69030b.png"/>
</p>

## Code


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
