import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import numpy as np

# 1. Tải mô hình ResNet-18 đã huấn luyện sẵn
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Sửa lỗi ở đây
model.eval()

# 2. Định nghĩa các bước xử lý ảnh đầu vào
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Tải danh sách nhãn ImageNet
try:
    with open("imagenet_classes.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file imagenet_classes.txt. Vui lòng tải file này từ https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a và đặt cùng thư mục với code.")

# Kiểm tra số lượng nhãn
if len(labels) != 1000:
    raise ValueError(f"File imagenet_classes.txt phải có đúng 1000 nhãn, nhưng chỉ tìm thấy {len(labels)} nhãn.")

# 4. Hàm dự đoán
def predict_image(image):
    if image is None:
        return None, "Vui lòng chọn một ảnh để nhận diện."

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
        results.append("\n Đây không phải là mèo!");

    return image, "\n".join(results)

# 5. Tạo giao diện Gradio
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Chọn ảnh để nhận diện"),
    outputs=[
        gr.Image(type="pil", label="Ảnh gốc"),
        gr.Textbox(label="Kết quả dự đoán")
    ],
    title="Nhận diện ảnh bằng ResNet-18",
    description="Tải lên một ảnh bất kỳ để nhận diện. Mô hình ResNet-18 sẽ dự đoán đối tượng trong ảnh (dựa trên 1000 lớp của ImageNet).",
)

# 6. Chạy ứng dụng
interface.launch()


# Resnet 50
def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL . Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D()(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
