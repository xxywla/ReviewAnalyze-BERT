from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

import config


def process():
    # 加载数据集
    dataset = load_dataset("csv", data_files=str(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv'))['train']

    print(dataset)
    # 过滤数据
    dataset = dataset.filter(lambda x: x['review'] is not None)
    print(dataset)

    # 设置标签
    dataset = dataset.cast_column('label', ClassLabel(names=['差评', '好评']))
    print(dataset)

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)

    # 处理数据集
    def tokenize(batch):
        # batch: {'review': [...], 'label': [...]}
        # __call__ 会返回一个包含 input_ids 和 attention_mask 的字典
        res = tokenizer(batch['review'], padding='max_length', truncation=True, max_length=config.SEQ_LEN)
        return {
            'input_ids': res['input_ids'],
            'attention_mask': res['attention_mask'],
            'label': batch['label']
        }

    dataset = dataset.map(tokenize, batched=True, remove_columns=['cat', 'review'])

    # 拆分 训练集 测试集
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    print(dataset_dict)

    # 保存处理后的数据
    dataset_dict['train'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'train'))
    dataset_dict['test'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'test'))

    print("数据处理完成！")


if __name__ == '__main__':
    process()
