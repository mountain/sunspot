import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(x)
else:
    print("MPS device not found.")
    device = torch.device("cpu")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import lightning.pytorch as pl
import numpy as np
import torch as th
from torch import optim, nn, utils

from sunspot.data.walks import Walks

total = 3253
epsilon = 1e-5

with open("data/points.txt") as f:
    lines = f.readlines()
points = th.tensor([[float(i) for i in line.split(',')] for line in lines], dtype=th.float32)[:, 1:]
if torch.backends.mps.is_available():
    points = points.to(device)


class HyperbolicEmbedding(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relations = nn.Parameter(th.stack([th.rand(24, 1), th.rand(24, 1) * 2 - 1], dim=1).to(device))
        self.encoder = nn.Sequential(nn.Linear(12, 256).to(device), nn.ReLU(), nn.Linear(256, 2).to(device))
        self.decoder = nn.Sequential(nn.Linear(2, 256).to(device), nn.ReLU(), nn.Linear(256, 12).to(device))
        if torch.backends.mps.is_available():
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            self.encoder = th.compile(self.encoder)
            self.decoder = th.compile(self.decoder)

    @th.compile
    def get_point_emb_batch(self, ix):
        xs, ys = points[ix, 0], points[ix, 1]
        ys = epsilon + th.abs(ys)
        return th.stack([xs, ys], dim=1)

    @th.compile
    def get_relation_emb_batch(self, ix):
        ix = (ix + 11) * (ix >= 0) + (ix + 12) * (ix < 0)
        rel = self.relations[ix]
        r = th.abs(rel[:, 0])
        theta = rel[:, 1] * np.pi * 2
        return th.stack([r * th.cos(theta), r * th.sin(theta)], dim=1)

    @th.compile
    def assigment(self, points):
        return - points[:, 0] / points[:, 1]

    @th.compile
    def flow(self, points, relations):
        return (self.assigment(points) + relations[:, 0]) * (1 + relations[:, 1])

    @th.compile
    def encode(self, batch):
        result = self.encoder(batch)
        return th.stack([result[:, 0], epsilon + th.abs(result[:, 1])], dim=1)

    @th.compile
    def decode(self, result):
        return self.decoder(result)

    @th.compile
    def recover(self, points):
        result = self.encoder(points)
        return self.decoder(result)

    def training_step(self, batch, batch_idx):
        prev, rel, loss = None, None, th.zeros(1)
        if torch.backends.mps.is_available():
            batch = batch.to(device)
            loss = loss.to(device)

        for ix in range(12):
            cur = self.encode(points[batch[:, 2 * ix]])
            if ix > 0:
                walk_error = (self.flow(prev, rel) - self.assigment(cur)) ** 2
                loss += th.mean(walk_error)
            prev = cur
            rel = self.get_relation_emb_batch(batch[:, 2 * ix + 1])

        for jx in range(12):
            point = self.get_point_emb_batch(batch[:, 2 * jx])
            linkj = self.get_relation_emb_batch(batch[:, 2 * jx + 1])
            length = (linkj[:, 0] ** 2 + linkj[:, 1] ** 2) / (epsilon + point[:, 1]) / (epsilon + point[:, 1])
            loss += th.mean(length)

        for kx in range(12):
            linkk = self.get_relation_emb_batch(batch[:, 2 * kx + 1])
            pointk = self.get_point_emb_batch(batch[:, 2 * kx])
            for lx in range(12):
                linkl = self.get_relation_emb_batch(batch[:, 2 * lx + 1])
                pointl = self.get_point_emb_batch(batch[:, 2 * lx])
                if (jx + kx) % 2 == 0:
                    link_error = + (linkk[:, 0] * linkl[:, 0] + linkk[:, 1] * linkl[:, 1]) / (epsilon + pointk[:, 1]) / (epsilon + pointl[:, 1])
                else:
                    link_error = - (linkk[:, 0] * linkl[:, 1] + linkk[:, 1] * linkl[:, 0]) / (epsilon + pointk[:, 1]) / (epsilon + pointl[:, 1])
                loss += th.mean(link_error)

        recover = self.recover(points)
        loss += th.mean((recover - points) ** 2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        import matplotlib
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt

        with th.no_grad():
            pt = self.encode(points).detach().cpu().numpy()
            x, y = pt[:, 0], pt[:, 1]
            plt.scatter(x, y, c=np.arange(total), cmap="viridis")
            self.logger.experiment.add_figure("emb", plt.gcf(), self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = Walks()
train_loader = utils.data.DataLoader(dataset, batch_size=256, num_workers=0, shuffle=True)
hypemb = HyperbolicEmbedding()
trainer = pl.Trainer(max_epochs=100, enable_progress_bar=True)
trainer.fit(model=hypemb, train_dataloaders=train_loader)
