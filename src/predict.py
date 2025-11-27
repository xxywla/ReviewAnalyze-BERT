import torch
from transformers import AutoTokenizer

import config
from model import ReviewAnalyzeModel


def predict_batch(input_ids, attention_mask, model):
    model.eval()
    with torch.no_grad():
        # input_ids shape is [batch_size, seq_len]

        output = model(input_ids, attention_mask)
        # output.shape is [batch_size]
        return torch.sigmoid(output).tolist()


def predict(text, model, tokenizer, device):
    tokenized = tokenizer([text], padding='max_length', truncation=True, max_length=config.SEQ_LEN, return_tensors='pt')

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    batch_result = predict_batch(input_ids, attention_mask, model)
    return batch_result[0]


def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODEL_FILE_NAME))

    print('请输入一段评论 q 或 quit 退出')
    while True:
        cur_input = input('> ')
        if cur_input == 'q' or cur_input == 'quit':
            break
        if cur_input.strip() == '':
            continue
        prob = predict(cur_input, model, tokenizer, device)
        if prob >= 0.5:
            print(f'正面评论，置信度: {prob:.4f}')
        else:
            print(f'负面评论，置信度: {1 - prob:.4f}')


if __name__ == '__main__':
    run_predict()
