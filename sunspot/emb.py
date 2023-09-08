import lightning.pytorch as pl
import torch as th
from torch import optim, nn, utils
from torchvision.transforms import ToTensor

from data.walks import Walks

total = 3253


class HyperbolicEmbedding(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.points = nn.Parameter(0.5 + th.rand(total, 2))
        self.relations = nn.Parameter(th.norm(th.zeros(24, 2), 1))

    def get_point_emb_batch(self, ix):
        return self.points[ix]

    def get_relation_emb_batch(self, ix):
        ix = (ix + 11) * (ix >= 0) + (ix + 12) * (ix < 0)
        return self.relations[ix]

    def training_step(self, batch, batch_idx):
        prev, rel, loss = None, None, th.zeros(1)
        for ix in range(12):
            cur = self.get_point_emb_batch(batch[2 * ix])
            if ix > 0:
                loss += th.sum(((prev[0] / prev[1] + rel[0]) * rel[1] - cur) ** 2)
            prev = cur
            rel = self.get_relation_emb_batch(batch[2 * ix + 1])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = Walks(transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)
hypemb = HyperbolicEmbedding()
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=hypemb, train_dataloaders=train_loader)