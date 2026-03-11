import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class PSNR(Metric):

    name = "psnr"

    def __init__(self, data_range: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = data_range
        self.metric = PeakSignalNoiseRatio(data_range=data_range)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        self.total += self.metric(pred, target)
        self.count += 1

    def compute(self) -> Tensor:
        return self.total / self.count


class SSIM(Metric):

    name = "ssim"

    def __init__(self, data_range: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = data_range
        self.metric = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        if pred.dim() == 3:
            pred = pred.unsqueeze(2)
            target = target.unsqueeze(2)
        self.total += self.metric(pred, target)
        self.count += 1

    def compute(self) -> Tensor:
        return self.total / self.count