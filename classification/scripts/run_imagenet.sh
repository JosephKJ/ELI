# 500 Base Classes Setting

# ImageNet - 5 task - iCaRL
python main.py --nb_cl_fg=500 --nb_cl=100 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_icarl_5_tasks.log

# ImageNet - 10 task - iCaRL
python main.py --nb_cl_fg=500 --nb_cl=50 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_icarl_10_tasks.log

# ImageNet - 25 task - iCaRL
python main.py --nb_cl_fg=500 --nb_cl=20 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_icarl_20_tasks.log


# LUCIR
# ImageNet - 5 task - LUCIR
python main.py --nb_cl_fg=500 --nb_cl=100 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_lucir_5_tasks.log

# ImageNet - 10 task - LUCIR
python main.py --nb_cl_fg=500 --nb_cl=50 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_lucir_10_tasks.log

# ImageNet - 25 task - LUCIR
python main.py --nb_cl_fg=500 --nb_cl=20 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_lucir_20_tasks.log

# AANET
# ImageNet - 5 task - AANET
python main.py --nb_cl_fg=500 --nb_cl=100 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_aanet_5_tasks.log

# ImageNet - 10 task - AANET
python main.py --nb_cl_fg=500 --nb_cl=50 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_aanet_10_tasks.log

# ImageNet - 25 task - AANET
python main.py --nb_cl_fg=500 --nb_cl=20 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --num_classes 1000 --ckpt_dir_fg=./base_500_imagenet.pt --resume_fg --data_dir=/proj/vot2021/x_fahkh/joseph/imagenet | tee ebm_imagenet_aanet_20_tasks.log
