

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
import paramiko

def download_model(remote_path, local_path, key_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ec2-18-218-205-80.us-east-2.compute.amazonaws.com', username='ubuntu', key_filename=key_path)

    sftp = ssh.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    ssh.close()

# Загрузка модели
remote_path = '/home/ubuntu/service/models/resnet50_modelmammae.pth'
local_path = 'resnet50_modelmammae.pth'  # Локальный путь, куда будет сохранена загруженная модель
key_path = '/Users/anna/folder2/vkr.pem'  # Путь к вашему ключу SSH на вашем локальном компьютере 

download_model(remote_path, local_path, key_path)

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
