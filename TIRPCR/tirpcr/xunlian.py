import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from model import LSTMModel  # 假设模型类名为MyModel
from dataset import Dataset  # 假设数据集类名为MyDataset

def main():
    # 参数配置
    data_dir = "D:\\Bshe\\Model\\DeeplPW\\DeepIPW-master\\text_preprocess\\save_cohort_all2023\\"
    pickles_dir = "D:\\Bshe\\Model\\DeeplPW\\DeepIPW-master\\text_preprocess\\pickles2023"

    model_params = {
        'diag_vocab_size': len(my_dataset.diag_code_vocab),  # 诊断词汇表大小
        'med_vocab_size': len(my_dataset.med_code_vocab),  # 药物词汇表大小
        'diag_embedding_size': args.diag_emb_size,  # 诊断嵌入尺寸
        'med_embedding_size': args.med_emb_size,  # 药物嵌入尺寸
        'diag_hidden_size': args.diag_hidden_size,  # 诊断隐藏层尺寸
        'med_hidden_size': args.med_hidden_size,  # 药物隐藏层尺寸
        'hidden_size': 100,  # 隐藏层尺寸（示例中为100，根据需要调整）
        'bidirectional': True,  # 是否使用双向LSTM
        'end_index': my_dataset.diag_code_vocab[CodeVocab.END_CODE],  # 结束索引
        'pad_index': my_dataset.diag_code_vocab[CodeVocab.PAD_CODE]  # 填充索引
    }

    model = LSTMModel(**model_params)
    if args.cuda:
        model = model.to('cuda')

    # 加载数据
    dataset = Dataset(data_dir, pickles_dir)
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split_train, split_val = int(0.7 * num_train), int(0.8 * num_train)
    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_val], indices[split_val:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # 初始化模型
    model = LSTMModel()
    if torch.cuda.is_available():
        model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for data in valid_loader:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {valid_loss / len(valid_loader):.4f}')

    # 评估
    model.eval()
    with torch.no_grad():
        # 计算指标如AUC, p值等
        pass

if __name__ == "__main__":
    main()
