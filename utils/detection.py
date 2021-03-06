
# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL
# TODO

# Import detectron2
import detectron2

# Import some common libraries
import numpy as np

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
        Tuple having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
    """
    # TODO
    outputs = DET_MODEL(img)

    # Step 1
    count_box = 0
    list_boxes = []

    for tens in outputs["instances"].pred_classes:
       if (tens == 7) or (tens == 2):
          list_boxes.append(outputs["instances"].pred_boxes[count_box])
       count_box = count_box + 1
   
    # Steps 2, 3 and 4
    if list_boxes == []:
      
       box_coordinates = (0, 0, img.shape[1], img.shape[0])

    else:

       max_area = 0
       box_coord_max_area = None

       for box in list_boxes:
          area = box.area()
          if area > max_area:
                max_area = area
                box_coord_max_area = box

       box_coordinates = (int(box_coord_max_area.tensor[0][0].item()), 
                      int(box_coord_max_area.tensor[0][1].item()), 
                      int(box_coord_max_area.tensor[0][2].item()), 
                      int(box_coord_max_area.tensor[0][3].item()))

    return box_coordinates
    
