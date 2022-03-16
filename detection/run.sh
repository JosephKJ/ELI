## Base 19
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/base_19.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
## 19 + 1
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/19_p_1.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
## 19 + 1 _ ft
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/ft_19_p_1.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
#sleep 10
# EBM Aligning: save the features
python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:52191' --config-file ./configs/PascalVOC-Detection/iOD/align_19_p_1.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
#sleep 10
# EBM Aligning: learn EBM and use it for aligning
python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:52190' --eval-only --config-file ./configs/PascalVOC-Detection/iOD/align_19_p_1.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 MODEL.ROI_HEADS.LEARN_INCREMENTALLY False


## Base 15
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/base_15.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
## 15 + 5
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/15_p_5.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
## 15 + 5 _ ft
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/ft_15_p_5.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
#sleep 10
# EBM Aligning: save the features
python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:52191' --config-file ./configs/PascalVOC-Detection/iOD/align_15_p_5.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
#sleep 10
# EBM Aligning: learn EBM and use it for aligning
python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:52190' --eval-only --config-file ./configs/PascalVOC-Detection/iOD/align_15_p_5.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 MODEL.ROI_HEADS.LEARN_INCREMENTALLY True


## Base 10
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/base_10.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
## 10 + 10
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/10_p_10.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
## 10 + 10 _ ft
#sleep 10
#python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/ft_10_p_10.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
#sleep 10
# EBM Aligning: save the features
python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:52191' --config-file ./configs/PascalVOC-Detection/iOD/align_10_p_10.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
#sleep 10
# EBM Aligning: learn EBM and use it for aligning
python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:52190' --eval-only --config-file ./configs/PascalVOC-Detection/iOD/align_10_p_10.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 MODEL.ROI_HEADS.LEARN_INCREMENTALLY False
