import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1. Tải mô hình ResNet-18 đã huấn luyện sẵn
model = models.resnet18(pretrained=True)
model.eval()  # Chuyển sang chế độ đánh giá (không huấn luyện)

# 2. Định nghĩa các bước xử lý ảnh đầu vào
# ResNet yêu cầu ảnh đầu vào 224x224 và chuẩn hóa theo ImageNet
preprocess = transforms.Compose([
    transforms.Resize(256),           # Phóng to ảnh lên 256x256
    transforms.CenterCrop(224),       # Cắt giữa thành 224x224
    transforms.ToTensor(),            # Chuyển thành tensor (0-1)
    transforms.Normalize(             # Chuẩn hóa theo ImageNet
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Tải và xử lý ảnh đầu vào
image_path = "cat2.jpg"  # Thay bằng đường dẫn ảnh của bạn
input_image = Image.open(image_path).convert("RGB")  # Mở ảnh và chuyển sang RGB
input_tensor = preprocess(input_image)  # Xử lý ảnh
input_batch = input_tensor.unsqueeze(0)  # Thêm chiều batch (1, 3, 224, 224)

# 4. Đưa ảnh qua mô hình để dự đoán
if torch.cuda.is_available():  # Dùng GPU nếu có
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():  # Không tính gradient để tiết kiệm tài nguyên
    output = model(input_batch)  # Đầu ra là tensor (1, 1000)

# 5. Chuyển đầu ra thành xác suất
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 6. Tải danh sách nhãn ImageNet để giải thích kết quả
with open("imagenet_classes.txt", "r") as f:  # Tải file nhãn (xem ghi chú dưới)
    labels = [line.strip() for line in f.readlines()]

# 7. Lấy top 5 dự đoán
top5_prob, top5_indices = torch.topk(probabilities, 5)
for i in range(5):
    print(f"Nhãn: {labels[top5_indices[i]]}, Xác suất: {top5_prob[i].item():.4f}")

# 8. In nhãn có xác suất cao nhất
top_prob, top_index = torch.max(probabilities, 0)
print(f"\nDự đoán cuối cùng: {labels[top_index]} với xác suất {top_prob.item():.4f}")