python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=2 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=cifar100 --resume_fg --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1012_153758 --evaluate_with_ebm | tee cifar_100_25_task_icarl.txt

python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=2 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=cifar100 --resume_fg --evaluate_with_ebm --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1012_014006 | tee cifar_100_25_task_lucir.txt

python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=2 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=free --branch_2=ss --dataset=cifar100 --resume_fg --evaluate_with_ebm --resume_with_ebm_training --ckpt_loc=./model_checkpoints/1027_114241 | tee cifar_100_25_task_aanet.txt
