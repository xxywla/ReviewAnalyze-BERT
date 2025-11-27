from datasets import load_from_disk
from torch.utils.data import DataLoader

import config


def get_dataloader(is_train=True):
    data_path = config.PROCESSED_DATA_DIR / ("train" if is_train else "test")
    train_dataset = load_from_disk(data_path)

    # 设置类型转换
    train_dataset.set_format(type='torch')

    return DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    print(f'train dataloader 个数: {len(train_dataloader)}')

    for batch in train_dataloader:
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['label'].shape)
        break
