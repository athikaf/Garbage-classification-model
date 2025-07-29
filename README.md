# ♻️ Smart Waste Classifier

An AI-powered image classification project that detects and classifies garbage types using deep learning models — **ResNet18** and **Vision Transformer (ViT)** — built with PyTorch, Streamlit, and timm. Designed to promote sustainability by aiding waste segregation.

## 🧠 Project Overview

This project was developed as part of an academic group assignment, focusing on applying deep learning to a real-world challenge: **automated waste classification**. The goal is to accurately classify garbage into categories like **metal**, **paper**, **plastic**, **glass**, **bio**, and **cardboard**, using two powerful computer vision models.

## 👥 Team Members

- **Ashwin Manickam Nagappan** – 101511260
- **Athika Fatima** – 101502209
- **Osemudiamen Iyamah** – 101511078
- **Mohammed Wali Uddin Qureshi** – 101589421
- **Agha Mohammed Hussain** – 101594440
- **Claude Sylvain Baumono Pugueu** – 101600567

## 📂 Project Structure

├── app.py # Streamlit app for interactive predictions
├── garbage.ipynb # Main training and evaluation notebook
├── vit_model.pth # Saved Vision Transformer model
├── resnet_model.pth # Saved ResNet18 model
├── val_labels.pth # True labels from validation set
├── vit_preds.pth # ViT model predictions
├── resnet_preds.pth # ResNet model predictions
├── requirements.txt # Python dependencies
└── README.md # You're here!

---

## 📊 Models Used

### ✅ ResNet18

A 18-layer deep residual network pretrained on ImageNet. Known for being lightweight and fast, it’s ideal for baseline comparisons and transfer learning tasks.

### ✅ ViT (Vision Transformer)

A transformer-based architecture that treats images as sequences of patches. More data-efficient and capable of modeling global context, but computationally heavier.

We trained both models for **10 epochs** using the same dataset to compare their performance on metrics like **precision**, **recall**, and **F1-score**.

---

## 🧹 Dataset

We used a cleaned version of the [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), organized into subfolders for each class. Each image was resized to **(256x256)** and normalized.

### Classes:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Bio

---

## 📈 Evaluation

Both models were evaluated using:

- **Confusion Matrix**
- **Precision, Recall, and F1-Score**
- **Class-wise performance comparison**
- **Visual comparison using matplotlib and seaborn**

Predicted labels and true labels were saved as `.pth` files for visualization and comparison.

---

## 🚀 Streamlit App

We created an interactive Streamlit web application (`app.py`) that allows users to:

- Upload or paste an image URL
- Choose between ResNet18 or ViT model
- View predictions with probability
- See model evaluation metrics (Precision, Recall, F1-Score)

### 🔧 How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/smart-waste-classifier.git
cd smart-waste-classifier

# Step 2: Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
streamlit run app.py

📚 Lessons Learned
Each team member brought unique strengths to the project:

Ashwin – Handled image preprocessing and dataset cleaning.

Athika – Led ViT model integration and Streamlit deployment.

Osemudiamen – Built evaluation visualizations and confusion matrix logic.

Mohammed – Worked on model training scripts and hyperparameter tuning.

Agha – Contributed to ResNet18 implementation and testing.

Claude – Assisted with Streamlit UI and prediction integration.

```

🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For significant changes, open an issue first to discuss what you'd like to change. For any questions or concerns, please feel free to reach out to <a href="https://www.linkedin.com/in/athika-fatima/">Athika Fatima</a>
