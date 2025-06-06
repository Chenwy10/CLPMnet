# Ultralytics YOLO ðŸš€, AGPL-3.0 license

from ultralytics.engine.results import Results, ResultsParsing
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.parsing_eval import evaluate_parsing
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import torch
import os
import cv2
import pdb

class ParsingPartMatchPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        #pdb.set_trace()
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "parsingpartmatch"
        self.pred_masks = []
        self.pred_labels = []
        self.pred_scores = []
        self.match_pairs = []
        self.image_lists = []
        self.images = []
        self.boxes = []

    def matchprocess(self, labels, node, scores):
        #pdb.set_trace()
        match_pair = []
        if node.dim() == 1:
            node = node.unsqueeze(0)
        node = F.normalize(node, dim=1)
        one_index = torch.where(labels == 1)[0]
        zero_index = torch.where(labels == 0)[0]
        if len(one_index)>0 and len(zero_index) > 0:
            #pdb.set_trace()
            node_one = node[one_index]
            node_zero = node[zero_index]
            final_map = torch.matmul(node_zero, node_one.T)
            max_index = torch.argmax(final_map, dim=0)
            for i in range(len(node_one)):
                if final_map[max_index[i]][i] > 0.5:
                    match_pair.append(torch.cat((zero_index[max_index[i]].unsqueeze(0), one_index[i].unsqueeze(0))))
        #pdb.set_trace()
        return torch.stack(match_pair)

    def postprocess(self, preds, img, orig_imgs):
        #pdb.set_trace()
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p, nms_indexs = ops.non_max_suppression_group(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
        #pdb.set_trace()
        results = []
        proto = preds[1][-3] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        pred_nodes = preds[1][-1].transpose(-1, -2)
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            
            pred_node = pred_nodes[i]
            nms_index = nms_indexs[i]
            nms_mask = nms_index.squeeze().int()
            pred_node = pred_node[nms_mask]
            
            #pdb.set_trace()
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask_parsing(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC  interpolation ???Ú´?
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                match_pair = self.matchprocess(pred[:, 5], pred_node, pred[:, 4])
            #results.append(ResultsParsing(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
            #pdb.set_trace()
            self.pred_masks.append(masks.cpu().detach())
            self.pred_labels.append(pred[:, 5].view(-1).cpu().detach())
            self.pred_scores.append(pred[:, 4].view(-1).cpu().detach())
            self.match_pairs.append(match_pair.cpu().detach())
            self.image_lists.append(img_path)
            self.images.append(orig_img)
            self.boxes.append(pred[:, :4])
        #pdb.set_trace()
        #return results
    
    def parsing_evaluation(self,):
        all_parsings = []
        all_scores = []
        all_segms = []
        #pdb.set_trace()
        for i in range(len(self.pred_masks)):
            #pdb.set_trace()
            pred_mask = [self.pred_masks[i][j].numpy() for j in range(len(self.pred_masks[i]))]
            pred_label = self.pred_labels[i].numpy()
            pred_score = self.pred_scores[i].numpy()
            all_bboxes = []
            all_score = []
            match_pair = np.vstack([self.match_pairs[i][j].numpy() for j in range(len(self.match_pairs[i]))])
            all_parsing = []
            segm = np.zeros((pred_mask[0].shape[0], pred_mask[0].shape[1], pred_mask[0].shape[2]), dtype=np.int32)
            zero_index = np.where(pred_label == 0)[0]
            for j in range(len(zero_index)):
                box = []
                score = []
                score.append(pred_score[zero_index[j]])
                box.append(self.boxes[i][zero_index[j]])
                mask = np.zeros((pred_mask[0].shape[1], pred_mask[0].shape[2]), dtype=np.int32)
                segm = (segm.astype(np.int32) | pred_mask[zero_index[j]].astype(np.int32)).astype(np.int32)
                for k in range(len(pred_mask[zero_index[j]])):
                    if k == 4:
                        mask[np.where(pred_mask[zero_index[j]][0] == 1)] = 1
                    else:
                        mask[np.where(pred_mask[zero_index[j]][k] == 1)] = k + 1
                indices = np.where(match_pair[:, 0] == zero_index[j])[0]
                for indice in indices:
                    one_index = match_pair[indice][1]
                    mask[np.where(pred_mask[one_index][4] == 1)] = 5
                    segm = (segm.astype(np.int32) | pred_mask[one_index].astype(np.int32)).astype(np.int32)
                    score.append(pred_score[one_index])
                    box.append(self.boxes[i][one_index])
                #pdb.set_trace()
                all_bboxes.append(box)
                all_parsing.append(mask)
                all_score.append(np.mean(score))
            #pdb.set_trace()
            all_segm = np.zeros((pred_mask[0].shape[1], pred_mask[0].shape[2]), dtype=np.int32)
            for k in range(len(segm)):
                if k == 0:
                    all_segm[np.where(segm[4] == 1)] = 5
                elif k == 4:
                    all_segm[np.where(segm[0] == 1)] = 1
                else:
                    all_segm[np.where(segm[k] == 1)] = k + 1
            #pdb.set_trace()
            all_parsings.append(all_parsing)
            all_segms.append(all_segm)
            all_scores.append(np.array(all_score))
            #self.save_image(os.path.basename(self.image_lists[i]), self.images[i], all_bboxes, all_parsing, all_score)
        #pdb.set_trace()
        evaluate_parsing(all_parsings, all_segms, all_scores, True, 0.001, 6, '/home/chenwy/ultralytics-github/datasets/Sperm_parsing_640_new/val_img/', '/home/chenwy/ultralytics-github/datasets/Sperm_parsing_640_new/annotations/val.json', self.image_lists, '/home/chenwy/ultralytics-github/datasets/Sperm_parsing_640_new/val_seg/')


    def save_image(self, img_name, img, all_bbox, all_result, all_score):
        #pdb.set_trace()
        mask_color_ori = [(220, 20, 60), (0, 202, 0), (212, 0, 228), (0, 160, 200), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                   (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                   (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                   (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                   (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                   (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                   (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                   (134, 134, 103), (145, 148, 174), (255, 208, 186),
                   (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                   (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                   (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
                   (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
                   (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
                   (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                   (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
                   (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
                   (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
                   (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
                   (191, 162, 208), (100, 60, 0), (100, 80, 0), (70, 0, 0),
                   (192, 0, 0), (30, 170, 250), (30, 170, 100), (0, 220, 220),
                   (175, 116, 175), (30, 0, 250), (42, 42, 165), (255, 77, 255),
                   (252, 226, 0), (255, 182, 182), (0, 82, 0), (157, 166, 120),
                   (0, 76, 110), (255, 57, 174), (0, 100, 199), (118, 0, 72),
                   (240, 179, 255), (92, 125, 0), (151, 0, 209), (182, 208, 188),
                   (176, 220, 0), (164, 99, 255), (73, 0, 92), (255, 129, 133),
                   (255, 180, 78), (0, 228, 0), (243, 255, 174), (255, 89, 45),
                   (103, 134, 143), (174, 148, 145), (186, 208, 255),
                   (255, 226, 197), (1, 134, 174), (54, 63, 109), (255, 138, 207),
                   (95, 0, 151), (61, 80, 9), (51, 105, 84), (105, 65, 74),
                   (102, 196, 164), (210, 195, 208), (65, 109, 255), (149, 143, 0),
                   (194, 0, 179), (106, 99, 209), (0, 121, 5), (205, 255, 227),
                   (208, 186, 147), (1, 69, 153), (161, 95, 3), (0, 255, 163),
                   (170, 0, 116), (199, 182, 0), (120, 165, 0), (88, 130, 183),
                   (0, 32, 95), (135, 114, 130), (133, 129, 110), (118, 74, 164),
                   (185, 142, 219), (114, 210, 79), (62, 90, 178), (15, 70, 65),
                   (115, 167, 127), (106, 105, 59), (45, 108, 142), (0, 172, 196),
                   (80, 54, 95), (255, 76, 128), (1, 57, 201), (112, 0, 246),
                   (208, 162, 191)]
        mask_color = []
        for tpl in mask_color_ori:
            lst = list(tpl)  # Convert the tuple to a list
            lst[0], lst[2] = lst[2], lst[0]  # Switch the first and third elements
            mask_color.append(tuple(lst)) 
        mask_color = np.array(mask_color, dtype=np.uint8) 
        #pdb.set_trace()
        for i in range(len(all_score)):
            for j in range(0,5):
                if j == 0:
                    color_mask = mask_color[4 + i*5]
                    mask = (all_result[i]==5)
                    img[mask] = color_mask
                elif j == 4:
                    color_mask = mask_color[0 + i*5]
                    mask = (all_result[i]==1)
                    img[mask] = color_mask
                else:
                    color_mask = mask_color[j + i*5]
                    mask = (all_result[i]==j+1)
                    img[mask] = color_mask
                        
            lines = []
            for j in range(len(all_bbox[i])):
                x1 = int(all_bbox[i][j][0])
                y1 = int(all_bbox[i][j][1])
                x2 = int(all_bbox[i][j][2])
                y2 = int(all_bbox[i][j][3])
                lines.append([(x1 + x2)//2, (y1 + y2)//2])
                #cv2.circle(img, ((x1 + x2)//2, (y1 + y2)//2), 3, (255, 0, 0), 3)
                cv2.rectangle(img,(x1, y1),(x2, y2),(0, 0, 255),2)
                    
            #if len(lines) > 1:
            #    for j in range(len(lines)):
            #        cv2.line(img, lines[0], lines[j], (0, 0, 255), 3)
        #pdb.set_trace()
        out_viz_file = os.path.join("./output640newmatchtest/", img_name)
        cv2.imwrite(out_viz_file, img)


    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                #pdb.set_trace()
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                #yield from self.results
            
        #self.run_callbacks("on_predict_end")
        self.parsing_evaluation()
        #evaluate_parsing()