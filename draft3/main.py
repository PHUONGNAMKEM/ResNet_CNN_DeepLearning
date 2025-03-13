import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import numpy as np

# Định nghĩa BasicBlock cho ResNet18
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

# Định nghĩa Bottleneck cho ResNet50
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.downsample(x)
        out = self.relu(out)
        return out

# Định nghĩa lớp ResNet tổng quát
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Hàm tạo ResNet18
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Hàm tạo ResNet50
def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Tạo mô hình (có thể chọn giữa ResNet18 hoặc ResNet50)
def get_model(model_name="resnet18"):
    if model_name is None:
        model_name = "resnet18"
    
    if model_name.lower() == "resnet18":
        model = resnet18(num_classes=1000)
        pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name.lower() == "resnet50":
        model = resnet50(num_classes=1000)
        pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Chỉ hỗ trợ 'resnet18' hoặc 'resnet50'")
    
    # Load pre-trained weights
    model.load_state_dict(pretrained.state_dict())
    model.eval()
    return model

# Xử lý ảnh đầu vào
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tải danh sách nhãn ImageNet
try:
    with open("imagenet_classes.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file imagenet_classes.txt.")

if len(labels) != 1000:
    raise ValueError(f"File imagenet_classes.txt phải có đúng 1000 nhãn, nhưng chỉ tìm thấy {len(labels)} nhãn.")

# Hàm dự đoán
def predict_image(image, model_name="resnet18"):
    if image is None:
        return None, "Vui lòng chọn một ảnh để nhận diện."
    
    if model_name is None:
        model_name = "resnet18"
    
    model = get_model(model_name)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_indices = torch.topk(probabilities, 5)
    results = []
    cat_prob_total = 0.0
    for i in range(5):
        idx = top5_indices[i].item()
        if idx < len(labels):
            label = labels[idx]
            prob = top5_prob[i].item()
            results.append(f"{label}: {prob:.4f}")
            if "cat" in label.lower():
                cat_prob_total += prob

    if cat_prob_total > 0.05:
        results.append(f"\nXác suất tổng là mèo: {cat_prob_total:.4f}")
        results.append("\nKết luận: Đây là mèo!")
    else:
        results.append(f"\nXác suất tổng là mèo: {cat_prob_total:.4f}")
        results.append("\nĐây không phải là mèo!")

    return image, "\n".join(results)

# Tạo giao diện Gradio
interface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Chọn ảnh để nhận diện"),
        gr.Dropdown(choices=["resnet18", "resnet50"], label="Chọn mô hình", value="resnet18")
    ],
    outputs=[
        gr.Image(type="pil", label="Ảnh gốc"),
        gr.Textbox(label="Kết quả dự đoán")
    ],
    title="Nhận diện ảnh bằng ResNet",
    description="Tải lên một ảnh và chọn mô hình (ResNet18 hoặc ResNet50) để nhận diện.",
)

# Chạy ứng dụng
interface.launch()