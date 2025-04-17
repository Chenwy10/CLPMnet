import sys
sys.path.append("/root/ultralytics")
from ultralytics import YOLO, YOLOPart, SpermParsingPart
import pdb
if __name__ == '__main__':
    model = SpermParsingPart('/home/chenwy/ultralytics-github/runs/best.pt', task='parsingpartmatch') 
    #pdb.set_trace()
    model.predict(
        source='/home/chenwy/ultralytics-github/datasets/Sperm_parsing_640_new_high_density/val_img/',
        save=True,  
        imgsz=1280, 
        conf=0.5,  
        iou=0.5,  
        show=False,  
        project='runs/predict',  
        name='exp',  
        save_txt=False, 
        save_conf=True,  
        save_crop=False, 
        show_labels=True,  
        show_conf=True,
        vid_stride=1,
        line_width=3,  
        visualize=False, 
        augment=False,  
        agnostic_nms=False, 
        retina_masks=False,  
        boxes=True,  
        device='1',
    )


