import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ===============================
# Utility functions
# ===============================
@st.cache_resource
def load_resnet_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_vit_model(num_classes):
    model = vit_b_16(pretrained=False)
    in_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("vit_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_image(image, model, classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, prediction = torch.max(outputs, 1)
    return classes[prediction.item()]

def show_metrics(y_true, y_pred, classes, model_name):
    st.subheader(f"📊 {model_name} Metrics")
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.text("Precision, Recall, F1 Score:")
    for cls in classes:
        st.text(f"{cls}: {report[cls]}")

# ===============================
# Main Streamlit App
# ===============================
st.title("♻️ Garbage Classifier App")
st.write("Classify waste using ResNet18 or Vision Transformer (ViT)")

model_choice = st.selectbox("Choose a Model", ["ResNet18", "ViT (Vision Transformer)"])

classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

if model_choice == "ResNet18":
    model = load_resnet_model(len(classes))
else:
    model = load_vit_model(len(classes))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        label = predict_image(img, model, classes)
        st.success(f"Prediction: **{label}**")

# ===============================
# Optional: Load Validation Metrics
# ===============================
if st.checkbox("Show Model Metrics (from validation set)"):
    try:
        y_true = torch.load("val_labels.pth")   # Save from notebook
        y_pred_resnet = torch.load("resnet_preds.pth")
        y_pred_vit = torch.load("vit_preds.pth")
        
        if model_choice == "ResNet18":
            show_metrics(y_true, y_pred_resnet, classes, "ResNet18")
        else:
            show_metrics(y_true, y_pred_vit, classes, "ViT")

    except FileNotFoundError:
        st.warning("Metrics not found. Make sure you've saved predictions and true labels from your notebook.")

