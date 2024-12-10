import torch
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from lightning.pytorch import LightningModule

from src.DRUID_layers import DRUID


class DRUIDSystem(LightningModule):
    def __init__(self, lr: float, N: int, nb_channels: int) -> None:
        super().__init__()

        self.model = DRUID(N, nb_channels)
        self.lr = lr

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, mask, gt = batch
        res = self.model(x, mask)

        return sum(mean_squared_error(gt, output) for output in res)

    def validation_step(self, batch: torch.Tensor) -> None:
        x, mask, gt = batch
        res = torch.clip(self.model(x, mask)[-1], 0, 1)

        self.log('Loss/Val', mean_squared_error(gt, res), prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int=0) -> None:
        x, mask, gt = batch
        res = torch.clip(self.model(x, mask)[-1], 0, 1)
        gt, res = gt[..., 2:-2, 2:-2], res[..., 2:-2, 2:-2]

        psnr = peak_signal_noise_ratio(gt, res, data_range=1, reduction=None, dim=(1, 2, 3))
        ssim = structural_similarity_index_measure(gt, res, data_range=1, reduction=None)

        self.log('Loss/Test_psnr', torch.mean(psnr))
        self.log('Loss/Test_psnr_std', torch.std(psnr))

        self.log('Loss/Test_ssim', torch.mean(ssim))
        self.log('Loss/Test_ssim_std', torch.std(ssim))

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int=0) -> torch.Tensor:
        x, mask, gt = batch
        res = torch.clip(self.model(x, mask), 0, 1)

        return torch.concat((gt[None], res))

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer],
                                            list[dict[str, torch.optim.lr_scheduler._LRScheduler]]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=1e-5)

        return [optimizer], [{'scheduler': scheduler, 'monitor': 'Loss/Val'}]
