import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import timm
import torch.nn as nn

# class ViTSmallClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(ViTSmallClassifier, self).__init__()
#         self.model = timm.create_model("vit_small_patch16_224", pretrained=False)
#         self.model.head = nn.Linear(self.model.head.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

class ViTTinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTTinyClassifier, self).__init__()
        self.model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# --- Wrapper Classes ---
class ResNetWrapper(nn.Module):
    def __init__(self, base_model):
        super(ResNetWrapper, self).__init__()
        self.network = base_model

    def forward(self, x):
        return self.network(x)

# class ViTWrapper(nn.Module):
#     def __init__(self, base_model):
#         super(ViTWrapper, self).__init__()
#         self.model = base_model

#     def forward(self, x):
#         return self.model(x)

# --- Load Models ---
@st.cache_resource
def load_resnet_model(num_classes):
    base_model = models.resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    model = ResNetWrapper(base_model)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_vit_model(num_classes):
    model = ViTTinyClassifier(num_classes)
    model.load_state_dict(torch.load("vit_model.pth", map_location="cpu"))
    model.eval()
    return model


# --- Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# --- Prediction ---
def predict(model, image_tensor, classes):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
    return classes[pred.item()]

# --- Metrics ---
def display_metrics(y_true, y_pred, model_name):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    st.markdown(f"### {model_name} Evaluation Metrics")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

# --- App UI ---
st.title("🗑️ Garbage Image Classifier")
model_choice = st.selectbox("Select a Model", ["ResNet18", "ViT-Small"])
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = load_resnet_model(len(classes)) if model_choice == "ResNet18" else load_vit_model(len(classes))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_tensor = preprocess_image(image)
        prediction = predict(model, image_tensor, classes)
        st.success(f"Predicted Class: **{prediction}**")

    except Exception as e:
        st.error(f"Image processing failed: {e}")

if st.checkbox("Show model evaluation"):
    try:
        y_true = torch.load("val_labels.pth").numpy()
        y_pred = torch.load("resnet_preds.pth" if model_choice == "ResNet18" else "vit_preds.pth").numpy()
        display_metrics(y_true, y_pred, model_choice)
    except Exception as e:
        st.error(f"Failed to load evaluation data: {e}")
