# Image-Classifier-MobileNetV2

Image Classifier — MobileNetV2 + Grad-CAM

A simple, lightweight image classification project built using MobileNetV2, fine-tuned on 5 flower categories, with Grad-CAM explainability.
Upload an image → get a prediction → see a heatmap showing what the model focused on.

Built entirely on a simple laptop, beginner-friendly, and fully open-source.

# 🔍 Features

-MobileNetV2 image classifier (fast + lightweight)

-5 flower classes: daisy, dandelion, rose, sunflower, tulip

-Grad-CAM heatmaps for explainability

-Clean and simple Gradio web interface

-CPU-friendly (no GPU required)

-Beginner-friendly project structure

# 🚀 Run the App
1. Install dependencies
pip install -r requirements.txt

2. Launch the Gradio interface
python app.py --model_path best_model.pth


It will open at:

http://127.0.0.1:7860


Upload a flower image → view predictions + Grad-CAM overlay.

# 🎓 Train the Model (Optional)

If you want to retrain:

python train.py --data_dir ./data --epochs 3 --batch_size 8 --output_model best_model.pth


Place your dataset inside the data/ folder.


# 🛠 Tech Stack

-Python

-PyTorch

-Torchvision

-Gradio

-NumPy

-Matplotlib

# ⭐ Why This Project?

This was built to understand:

-Transfer Learning

-Image Classification

-Grad-CAM Explainability

-Building AI apps on a low-end laptop

-Deploying simple ML demos


## Built by Alok<3
