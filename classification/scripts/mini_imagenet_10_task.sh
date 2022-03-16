# 10 Task iCaRL
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=imagenet_sub --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --ckpt_dir_fg=./base_50_imagenet_sub.pt --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1026_045637 | tee imagenet_subset_10_task_icarl.log

# 10 Task LUCIR
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=imagenet_sub --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --ckpt_dir_fg=./base_50_imagenet_sub.pt --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1026_050528 | tee imagenet_subset_10_task_lucir.log

python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --evaluate_with_ebm --ebm_latent_dim 512 --num_workers 16 --epochs 90 --custom_weight_decay 1e-4 --ebm_n_layers 0 --ckpt_dir_fg=./base_50_imagenet_sub.pt --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1105_033045 | tee imagenet_subset_10_task_aanet.log
