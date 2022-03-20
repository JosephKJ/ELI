# Energy-based Latent Aligner for Incremental Learning: Object Detection
### CVPR 2022

## Installation and setup
- Install the Detectron2 library that is packages along with this code base. See [INSTALL.md](INSTALL.md).
- Download and extract Pascal VOC 2007 to `./datasets/VOC2007/`
- ELI enhances already trained incremental models. You can download trained incremental models from [iOD repository](https://github.com/JosephKJ/iOD). Here are the direct links: [19+1](https://drive.google.com/file/d/1pocjYPenjXda0fRh7ir_c1ItyAZCBoEN/view?usp=sharing) [15+5](https://drive.google.com/file/d/1KaynMWxb6nHytfMYP_wh8Dy-AvsLLazQ/view?usp=sharing) [10+10](https://drive.google.com/file/d/1aWc-1P7ZtNrye_asN5mKMtu7G8G0tLAm/view?usp=sharing)
- Use the script: `run.sh` (You may uncomment the other commands in this scrips to train the base and incremetal models.)


##### Our environment configurations
- Python version: 3.6.7
- PyTorch version: 1.3.0
- CUDA version: 11.0
- GPUs: 4 x NVIDIA GTX 1080-ti

## Acknowledgement
The code is build on top of Detectron2 library. 

## Citation
If you find our research useful, please consider citing us:

```BibTeX

@inproceedings{joseph2022Energy,
  title={Energy-based Latent Aligner for Incremental Learning},
  author={Joseph, KJ and Khan, Salman and Khan, Fahad Shahbaz and Anwar, Rao Muhammad and Balasubramanian, Vineeth},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{joseph2021incremental,
  title={Incremental object detection via meta-learning},
  author={Joseph, KJ and Rajasegaran, Jathushan and Khan, Salman and Khan, Fahad Shahbaz and Balasubramanian, Vineeth},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
