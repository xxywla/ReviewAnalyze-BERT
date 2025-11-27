import torch
from tqdm import tqdm

from dataset import get_dataloader
import config
from predict import predict_batch
from model import ReviewAnalyzeModel


def evaluate(model, dataloader, device):
    total_num, acc_num = 0, 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["label"].tolist()

        predicts = predict_batch(input_ids, attention_mask, model)
        for predict, target in zip(predicts, targets):
            pred_label = 1 if predict > 0.5 else 0
            if pred_label == target:
                acc_num += 1
            total_num += 1
    return acc_num / total_num


def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader(False)

    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODEL_FILE_NAME))

    accuracy = evaluate(model, dataloader, device)

    print(f'准确率: {accuracy:.4f}')


if __name__ == '__main__':
    run_evaluate()
