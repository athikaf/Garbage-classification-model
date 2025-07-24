import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Load class names (ensure this matches your training dataset order)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Define model
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = ResNet18Classifier(len(classes))
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Title
st.title("🗑 Garbage Classifier")
st.markdown("Paste an image URL and this model will predict the type of garbage.")

# Input
img_url = st.text_input("Enter image URL:")

if img_url:
    try:
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(image, caption="Input Image", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(img_tensor)
            _, prediction = torch.max(outputs, dim=1)
            pred_class = classes[prediction.item()]

        st.success(f"Predicted Garbage Category: **{pred_class}**")

    except Exception as e:
        st.error(f"Could not process image. Error: {e}")
