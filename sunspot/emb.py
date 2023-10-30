import lightning.pytorch as pl
import numpy as np
import torch as th
from torch import optim, nn, utils

from sunspot.data.walks import Walks

total = 3253

with open("data/points.txt") as f:
    lines = f.readlines()
points = th.tensor([[float(i) for i in line.split(',')] for line in lines], dtype=th.float32)[:, 1:]


class HyperbolicEmbedding(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relations = nn.Parameter(th.stack([th.rand(24, 1), th.rand(24, 1) * 2 - 1], dim=1))
        self.encoder = nn.Sequential(nn.Linear(12, 256), nn.ReLU(), nn.Linear(256, 2))

    def get_point_emb_batch(self, ix):
        return self.points[ix]

    def get_relation_emb_batch(self, ix):
        ix = (ix + 11) * (ix >= 0) + (ix + 12) * (ix < 0)
        rel = self.relations[ix]
        r = rel[:, 0] * rel[:, 0]
        theta = rel[:, 1] * np.pi
        return th.stack([r * th.cos(theta), r * th.sin(theta)], dim=1)

    def assigment(self, points):
        return - points[:, 0] / points[:, 1]

    def flow(self, points, relations):
        return (self.assigment(points) + relations[:, 0]) * (1 + relations[:, 1])

    def encode(self, batch):
        result = self.encoder(batch)
        return th.stack([result[:, 0], 100000 * th.sigmoid(result[:, 1])], dim=1)

    def training_step(self, batch, batch_idx):
        prev, rel, loss = None, None, th.zeros(1)
        for ix in range(12):
            cur = self.encode(points[batch[:, 2 * ix]])
            if ix > 0:
                walk_error = (self.flow(prev, rel) - self.assigment(cur)) ** 2
                loss += th.mean(walk_error)
            prev = cur
            rel = self.get_relation_emb_batch(batch[:, 2 * ix + 1])
        for jx in range(12):
            linkj = self.get_relation_emb_batch(batch[:, 2 * jx + 1])
            # length = (linkj[:, 0] ** 2 + linkj[:, 1] ** 2)
            # loss += th.mean(length) / 144
            for kx in range(12):
                linkk = self.get_relation_emb_batch(batch[:, 2 * kx + 1])
                if (jx + kx) % 2 == 0:
                    link_error = + (linkj[:, 0] * linkk[:, 0] + linkj[:, 1] * linkk[:, 1])
                else:
                    link_error = - (linkj[:, 0] * linkk[:, 1] + linkj[:, 1] * linkk[:, 0])
                loss += th.mean(link_error) / 144
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        import matplotlib
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt

        with th.no_grad():
            pt = self.encode(points).detach().numpy()
            x, y = pt[:, 0], pt[:, 1]
            plt.scatter(x, y, c=np.arange(total), cmap="viridis")
            self.logger.experiment.add_figure("emb", plt.gcf(), self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = Walks()
train_loader = utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
hypemb = HyperbolicEmbedding()
trainer = pl.Trainer(max_epochs=1000, enable_progress_bar=True)
trainer.fit(model=hypemb, train_dataloaders=train_loader)
