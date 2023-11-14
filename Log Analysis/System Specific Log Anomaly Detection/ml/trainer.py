import numpy as np
import torch
import time


class SimpleLossCompute:

    def __init__(self, model, criterion, opt=None, is_test=False):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.is_test = is_test

    def __call__(self, x, y, dist):
        loss = torch.mean((1 - y) * torch.sqrt(dist) - (y) * torch.log(1 - torch.exp(-torch.sqrt(dist))))

        if not self.is_test:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item()


def run_train(dataloader, model, loss_compute, step_size=10):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(dataloader):

        b_input, b_labels = batch

        out = model.forward(b_input.cuda(), b_labels.cuda(), None, None)
        dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
        loss = loss_compute(out, b_labels.cuda(), dist)
        total_loss += loss

        if i % step_size == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d / %d Loss: %f" %
                  (i, len(dataloader), loss))
            start = time.time()
            tokens = 0
    return total_loss


def run_test(dataloader, model, loss_compute, step_size=10):
    preds = []
    distances = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b_input, b_labels = batch

            out = model.forward(b_input.cuda(), b_labels.cuda(),
                                None, None)
            out_p = model.generator(out)
            dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
            loss = loss_compute(out, b_labels.cuda(), dist)
            if i % step_size == 1:
                print("Epoch Step: %d / %d Loss: %f" %
                      (i, len(dataloader), loss))
            tmp = out_p.cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
            distances += list(dist.cpu().numpy())

    return preds, distances
