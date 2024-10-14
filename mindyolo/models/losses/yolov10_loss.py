from mindspore import nn, ops

from .yolov8_loss import YOLOv8Loss
from mindyolo.models.registry import register_model

__all__ = ["YOLOv10Loss"]

@register_model
class YOLOv10Loss(nn.Cell):
    def __init__(self):
        self.one2many = YOLOv8Loss(tal_topk=10)
        self.one2one = YOLOv8Loss(tal_topk=1)

    def construct(self, preds, batch):
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], ops.concat(loss_one2many[1], loss_one2one[1])