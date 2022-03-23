# ELI for Image Classification (CVPR 2022)

## Setup

### Prerequisite
- Python version: 3.6.7
- PyTorch version: 1.3.0

### Data
- CIFAR-100: Automatically downloaded in the code.
- ImageNet-subset: Please download from [here](https://drive.google.com/file/d/1n5Xg7Iye_wkzVKc0MTBao5adhYSUlMCL/view?usp=sharing). Untar the contents into `data/seed_1993_subset_100_imagenet`
- ImageNet: Please download from [official website](https://www.image-net.org/download.php), setup the `val` directory following [this](https://github.com/soumith/imagenet-multiGPU.torch#data-processing), and update `--data_dir` argument to your data path.

### Models
ELI enhances trained models. We share our base models [here](https://drive.google.com/file/d/1aqVkruS1oKKmqAcWHumbbPRB9TP_w1Ls/view?usp=sharing). Download and extract to `./model_checkpoints`.

### Alignign using ELI
Please use these script to align the latents with ELI:
- [cifar_100_5_task.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/cifar_100_5_task.sh)
- [cifar_100_10_task.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/cifar_100_10_task.sh)
- [cifar_100_25_task.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/cifar_100_25_task.sh)
- [mini_imagenet_5_task.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/mini_imagenet_5_task.sh)
- [mini_imagenet_10_task.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/mini_imagenet_10_task.sh)
- [mini_imagenet_25_task.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/mini_imagenet_25_task.sh)
- [run_imagenet.sh](https://github.com/JosephKJ/ELI/tree/main/classification/scripts/run_imagenet.sh)

## Acknowledgement
Our code is build on top of the [AANET](https://github.com/yaoyao-liu/class-incremental-learning/tree/main/adaptive-aggregation-networks) code base. Please consider citing their paper too, if you find our work useful. 

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
