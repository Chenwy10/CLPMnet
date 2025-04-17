# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.data import build_dataloader, build_yolo_dataset, build_parsing_part_match_dataset
from ultralytics.nn.tasks import ParsingPartMatchModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, check_segpart_dataset
from ultralytics.utils import emojis
from ultralytics.utils.torch_utils import de_parallel
import pdb

class ParsingPartMatchTrainer(yolo.detect.DetectionTrainer):
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
        overrides["task"] = "parsingpartmatch"
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
        return build_parsing_part_match_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    
    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        #pdb.set_trace()
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        self.model.namespart = self.data["namespart"]
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        #pdb.set_trace()
        model = ParsingPartMatchModel(cfg, ch=3, nc=self.data["nc"], npc=self.data["npc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

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
            elif self.args.task == "parsingpartmatch":
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
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "glb_seg_loss", "match_loss"
        return yolo.parsingpartmatch.ParsingPartMatchValidator(
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
