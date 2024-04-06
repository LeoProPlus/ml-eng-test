import torch
from tqdm import tqdm
from ml.walls_detection.checkpoint import save_checkpoint


class Trainer:
    def train(epochs, model, train_data_loader, valid_data_loader, optimizer, device):
        all_train_losses = []
        all_val_losses = []

        for epoch in range(1, epochs+1):
            train_loss, val_loss = Trainer.run_epoch(
                model, train_data_loader, valid_data_loader, optimizer, device)

            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)

            save_checkpoint(model, optimizer, epoch, train_loss, val_loss)

            print('epoch:', epoch, "  train_loss: ",
                  train_loss, "  val_loss: ", val_loss)

        history = {
            'train_losses': all_train_losses,
            'val_losses': all_val_losses
        }

        return history

    def run_epoch(model, train_data_loader, valid_data_loader, optimizer, device):
        # Set the model to training mode
        model.train()

        train_epoch_loss = 0
        val_epoch_loss = 0
        for i, (imgs, targ) in enumerate(tqdm(train_data_loader)):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs, targets)
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        with torch.no_grad():
            for j, (imgs, targ) in enumerate(tqdm(valid_data_loader)):
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targ]
                loss = model(imgs, targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()

        return train_epoch_loss, val_epoch_loss
