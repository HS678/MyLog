# 数据准备
import torch
from torchvision import transforms
from torchtext.data.utils import get_tokenizer
from PIL import Image

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 文本预处理
tokenizer = get_tokenizer("basic_english")

# 加载图像和文本
image = Image.open("example.jpg")
text = "A beautiful sunset over the mountains."

# 预处理图像
image_tensor = transform(image)

# 预处理文本
text_tokens = tokenizer(text)


# 特征提取
from torchvision.models import resnet50
from torchtext.vocab import GloVe

# 图像特征提取
image_model = resnet50(pretrained=True)
image_model.eval()
image_features = image_model(image_tensor.unsqueeze(0))

# 文本特征提取
glove = GloVe(name='6B',dim=100)
text_features = torch.stack([glove[token] for token in text_tokens]).mean(dim=0)


# 模态融合
combined_features = torch.cat((image_features, text_features.unsqueeze(0)), dim=1)

# 模型训练
import torch.nn as nn
import torch.optim as optim

class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    

# 定义模型
model = MultimodalClassifier(input_dim=combined_features.size(1), num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(combined_features)
    loss = criterion(outputs, torch.tensor([0]))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss:{loss.item()}")