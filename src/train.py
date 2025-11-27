import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import ReviewAnalyzeModel


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()

    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["label"].to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ReviewAnalyzeModel().to(device)

    dataloader = get_dataloader()

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        print(f'=====Epoch {epoch + 1}/{config.EPOCHS}=====')
        epoch_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)

        writer.add_scalar('Loss', epoch_loss, epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), config.MODEL_FILE_NAME)
            print('模型保存成功')
        else:
            print('无需保存')

        print(f'Epoch {epoch + 1}/{config.EPOCHS}, Loss: {epoch_loss}')


if __name__ == '__main__':
    train()
