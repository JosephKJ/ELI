# 25 Task iCaRL
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=imagenet_sub --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --ckpt_dir_fg=./base_50_imagenet_sub.pt --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1026_050900 | tee imagenet_subset_25_task_icarl.log

# 25 Task LUCIR
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=imagenet_sub --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --ckpt_dir_fg=./base_50_imagenet_sub.pt --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1114_025033 | tee imagenet_subset_25_task_lucir.log

# 25 Task AANET
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --ebm_n_layers 0 --ckpt_dir_fg=./base_50_imagenet_sub.pt --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1105_033050 | tee imagenet_subset_25_task_aanet.log
