# 输入数据
import torch


x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# 模型参数
w = torch.tensor([[1.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

# 前向传播
y_pred = x @ w + b

# 计算损失
loss = ((y_pred - y) ** 2).mean()
print(loss)