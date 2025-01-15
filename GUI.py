import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

# 加载训练好的模型
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except RuntimeError as e:
        raise ValueError(f"Model loading failed. Check class count and model structure: {e}")
    model = model.to(device)
    model.eval()
    return model

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# 进行分类预测
def predict(image_path, model, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
    return class_names[predicted_class], confidence.item()

# 创建 GUI 界面
def create_gui(model, class_names):
    def open_file():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            label_result.config(text="Processing...")
            try:
                predicted_class, confidence = predict(file_path, model, class_names)
                label_result.config(text=f"Class: {predicted_class}\nConfidence: {confidence:.2f}")
                img = Image.open(file_path).resize((200, 200))
                img = ImageTk.PhotoImage(img)
                label_image.config(image=img)
                label_image.image = img
            except Exception as e:
                label_result.config(text=f"Error: {str(e)}")

    # 创建主窗口
    root = tk.Tk()
    root.title("Image Classification")

    # 创建按钮和标签
    btn_open = Button(root, text="Select Image", command=open_file)
    btn_open.pack()

    label_image = Label(root)
    label_image.pack()

    label_result = Label(root, text="", font=("Helvetica", 14))
    label_result.pack()

    # 运行 GUI 循环
    root.mainloop()

if __name__ == "__main__":
    model_path = "./best_model.pth"  # 模型路径
    # 替换为训练模型时使用的类别名称
    dataset_dir = "./Vegetable Images/train"
    if os.path.exists(dataset_dir):
        class_names = sorted(os.listdir(dataset_dir))
    else:
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    num_classes = len(class_names)

    model = load_model(model_path, num_classes)
    create_gui(model, class_names)
