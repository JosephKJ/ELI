import cv2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Get image
im = cv2.imread("/home/joseph/workspace/eb-iOD/datasets/VOC2007/JPEGImages/000112.jpg")

# Get the configuration ready
cfg = get_cfg()
cfg.merge_from_file("configs/PascalVOC-Detection/iOD/align_15_p_5.yaml")
# cfg.MODEL.WEIGHTS = "/home/joseph/PycharmProjects/detectron2/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)

v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img = v.get_image()[:, :, ::-1]
cv2.imwrite('output.jpg', img)

