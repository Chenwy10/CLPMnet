# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.data import build_dataloader, build_yolo_dataset, build_parsing_part_dataset, build_parsing_part_WBC_dataset
from ultralytics.nn.tasks import ParsingPartModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, check_segpart_dataset
from ultralytics.utils import emojis
from ultralytics.utils.torch_utils import de_parallel
import torch
import pdb

class ParsingPartWBCTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        #pdb.set_trace()
        if overrides is None:
            overrides = {}
        overrides["task"] = "parsingpartWBC"
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        #pdb.set_trace()
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_parsing_part_WBC_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = ParsingPartModel(cfg, ch=3, nc=self.data["nc"], npc=self.data["npc"], verbose=verbose and RANK == -1)
        #pdb.set_trace()
        if weights:
            model.load(weights)
        
        if 'pretrained' in cfg and not weights:
            model = self.get_backbone(model, cfg['pretrained'])
        
        return model

    #def get_backbone(self, model, backbone):
    #    pdb.set_trace()
    #    from torchvision import models
        
        #if backbone == 'resnet50':
        #    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Copy weights from the pretrained model to the manually defined model
    #    pretrained_dict = torch.load(backbone)['state_dict']
    #    pretrained_dict = pretrained_model.state_dict()
    #    model_dict = model.state_dict()
        
        #pdb.set_trace()
        
    #    for my_key, resnet_key in zip(model_dict.keys(), pretrained_dict.keys()):
    #        if 'fc' in resnet_key:
                #pdb.set_trace()
    #            break
    #        model_dict[my_key] = pretrained_dict[resnet_key]
        
        #pdb.set_trace()
        
        # Load the updated state dict into the manually defined model
    #    model.load_state_dict(model_dict)
        
    #    return model
        
    def get_backbone(self, model, backbone):
        #pdb.set_trace()
        #from torchvision import models
        
        #if backbone == 'resnet50':
        #    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Copy weights from the pretrained model to the manually defined model
        pretrained_dict = torch.load(backbone)['model'].state_dict()
        #pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        #pdb.set_trace()
        
        for model_key, pretrained_key in zip(model_dict.keys(), pretrained_dict.keys()):
            parts = model_key.split('.')
            if int(parts[1]) > 9:
                #pdb.set_trace()
                break
            model_dict[model_key] = pretrained_dict[pretrained_key]
        
        #pdb.set_trace()
        
        # Load the updated state dict into the manually defined model
        model.load_state_dict(model_dict)
        
        return model
        
    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        #pdb.set_trace()
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.task == "parsingpartWBC":
                data = check_segpart_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ? {e}")) from e
        self.data = data
        #pdb.set_trace()
        return data["train"], data.get("val") or data.get("test")

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "glb_seg_loss"
        return yolo.parsingpartWBC.ParsingPartWBCValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        #pdb.set_trace()
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
