from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import json

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class Predictor:
    def __init__(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13 
        cfg.MODEL.WEIGHTS = "predict/model/clothe_seg.pth" # model이 저장된 경로
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set the testing threshold for this model
        cfg.DATASETS.TEST = ("DeepFashion",)
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img):
        np_array = np.frombuffer(img.read(), dtype=np.uint8)
        im = cv2.imdecode(np_array, -1)
                
        outputs = self.predictor(im)
        instances = outputs["instances"]

        result_array = []

        for i in range(len(instances)):
            score = instances.scores[i]

            if score >= 0.8:
                pred_class = int(instances.pred_classes[i])
                segmentation_mask = instances.pred_masks[i].cpu().numpy()

                rows, cols = np.where(segmentation_mask)
                coordinates = [[int(x), int(y)] for x, y in zip(cols, rows)]

                object_dict = {
                    "type": pred_class,
                    "coordinates": coordinates
                }

                print(pred_class)
                result_array.append(object_dict)

        return result_array
