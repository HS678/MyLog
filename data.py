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