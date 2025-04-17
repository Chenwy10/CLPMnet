import sys
sys.path.append("/root/ultralytics")
from ultralytics import YOLO, YOLOPart, SpermParsingPart

if __name__ == '__main__':
    model = SpermParsingPart('/home/chenwy/ultralytics-github/runs/WBC_best.pt', task='parsingpartWBC') 
    model.predict(
        source='/home/chenwy/ultralytics-github/datasets/WBC_new/val_img/',
        save=True,  
        imgsz=1024, 
        conf=0.05,  
        iou=0.5,  
        show=False,  
        project='runs_new/predict',  
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
        device='3',
    )


