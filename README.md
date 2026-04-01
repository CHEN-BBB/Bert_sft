# Bert_SFT

## 项目描述

这是一个使用 BERT 模型进行自然语言推断（Natural Language Inference）任务的项目。通过微调预训练的 BERT 模型，我们可以在 SNLI 数据集上进行文本对分类，包括判断前提和假设之间的关系：蕴涵（entailment）、矛盾（contradiction）或中性（neutral）。

项目基于 Dive into Deep Learning (d2l) 库实现，使用 PyTorch 作为深度学习框架。

## 安装步骤

### 环境要求

- Python 3.7+
- PyTorch
- CUDA（可选，用于 GPU 加速）

### 安装依赖

1. 克隆项目仓库：
   ```bash
   git clone <repository-url>
   cd Bert_sft
   ```

2. 安装必要的 Python 包：
   ```bash
   pip install d2l==0.16
   ```

   这将自动安装所需的依赖，包括 PyTorch、torchvision 等。

3. （可选）如果需要 GPU 支持，请确保安装了正确的 PyTorch 版本：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 使用示例

### 数据准备

项目使用 SNLI 数据集。运行 notebook 时，d2l 库会自动下载和处理数据。

### 训练模型

1. 打开 `natural-language-inference-bert.ipynb` notebook。

2. 运行所有细胞，按照顺序执行：
   - 安装依赖
   - 导入库
   - 加载预训练 BERT 模型
   - 准备数据集
   - 定义分类器
   - 训练模型

3. 训练参数可以调整：
   - `batch_size`: 批大小（默认 512）
   - `max_len`: 最大序列长度（默认 128）
   - `num_epochs`: 训练轮数（默认 5）
   - `lr`: 学习率（默认 1e-4）

### 预测示例

训练完成后，可以使用模型进行预测：

```python
# 示例预测代码
sample_idx = 0
(tokens_X, segments_X, valid_lens_x), true_label_idx = test_set[sample_idx]

# 模型预测
net.eval()
with torch.no_grad():
    predictions = net((tokens_X.unsqueeze(0).to(devices[0]),
                       segments_X.unsqueeze(0).to(devices[0]),
                       valid_lens_x.unsqueeze(0).to(devices[0])))
    predicted_label = torch.argmax(predictions, dim=1).item()

print(f"预测结果: {label_map[predicted_label]}")
print(f"真实标签: {label_map[true_label_idx.item()]}")
```

## 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

### 报告问题

- 使用 GitHub Issues 报告 bug 或请求新功能
- 提供详细的描述，包括错误信息、环境信息和重现步骤

### 提交代码

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature-name`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature-name`
5. 提交 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码规范
- 为新功能添加适当的文档和注释
- 确保所有测试通过

### 测试

- 在提交前运行 notebook 中的所有细胞，确保没有错误
- 测试模型在不同数据集上的性能

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 参考

- [Dive into Deep Learning](https://d2l.ai/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)