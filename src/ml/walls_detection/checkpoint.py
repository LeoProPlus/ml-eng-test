import torch
import os


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss):
    dir = 'checkpoints/tmp'
    file_name = f'model_{epoch}_{val_loss:.5f}.pt'
    path = os.path.join(dir, file_name)

    os.makedirs(dir, exist_ok=True)

    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    torch.save(data, path)

    return path


def laod_checkpoint(path, model, optimizer):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    return epoch, train_loss, val_loss


def laod_model_checkpoint(path, model):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from file: {path}')
