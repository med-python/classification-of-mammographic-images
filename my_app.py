import streamlit as st
import os
from werkzeug.utils import secure_filename
import pydicom
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

# Функция классификации DICOM файла
def classify_dicom(filepath):
    # Функция нормализации и визуализации DICOM
    def normalize_visualize_dicom_1(dcm_file):
        dicom_file = pydicom.dcmread(dcm_file)
        dicom_array = dicom_file.pixel_array.astype(float)
        normalized_dicom_array = ((np.maximum(dicom_array, 0))/dicom_array.max()) * 255.0
        uint8_image = np.uint8(normalized_dicom_array)
        return uint8_image

    # Классификация DICOM файла
    image_1 = normalize_visualize_dicom_1(filepath)
    if image_1 is not None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.fromarray(image_1)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        resnet50.eval()
        with torch.no_grad():
            output = resnet50(img_tensor.to(device))
            predicted_class = torch.round(output).item()

        if predicted_class == 1:
            return "Изображение классифицировано как рак."
        else:
            return "Изображение классифицировано как здоровое."

# Загрузка предварительно обученной модели ResNet50
resnet50 = models.resnet50(pretrained=True)

num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

# Изменение пути к модели
checkpoint_path = 'models/resnet50_modelmammae.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    resnet50.load_state_dict(checkpoint['model_state_dict'])
else:
    resnet50.load_state_dict(checkpoint)

def main():
    st.title('DICOM Классификация')
    uploaded_file = st.sidebar.file_uploader("Загрузите файл DICOM", type=['dcm'])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        filepath = os.path.join('uploads', secure_filename(uploaded_file.name))
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = classify_dicom(filepath)
        st.write(result)
if __name__ == '__main__':
    main()
    