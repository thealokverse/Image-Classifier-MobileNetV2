# app.py
# Gradio app: load model, run inference on uploaded image, compute Grad-CAM, show overlay & explanation.
# Usage:
#  python app.py --model_path best_model.pth
# built by @thealokverse on X
# cool shit

import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import gradio as gr

#Simple Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer_name="features"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        # register hook to the last conv layer of mobilenet_v2: model.features[-1] is usually conv
        target = None
        # try to find last conv-like module under features
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                target = module
        # fallback: use the last module of model.features
        if target is None:
            target = self.model.features[-1]
        # forward hook
        def forward_hook(module, input, output):
            self.activations = output.detach()
        # backward hook
        def backward_hook(module, grad_in, grad_out):
            # grad_out can be a tuple; take the first tensor
            self.gradients = grad_out[0].detach()
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        # forward
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        # backward
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        weights = grads.mean(dim=(2,3), keepdim=True)  # global avg pool
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        # normalize to [0,1]
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() != 0:
            cam_np = cam_np / cam_np.max()
        return cam_np, class_idx

#Helpers
def load_model(model_path=None, device=torch.device("cpu")):
    # If user saved model (train.py), that contains classes & state dict
    if model_path and Path(model_path).exists():
        print("Loading saved model:", model_path)
        ckpt = torch.load(model_path, map_location=device)
        classes = ckpt.get('classes', None)
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_features, len(classes)))
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        return model, classes
    else:
        # Use ImageNet pretrained model (class names are ImageNet labels)
        print("No saved model found — using ImageNet model.")
        model = models.mobilenet_v2(pretrained=True)
        # load imagenet labels
        try:
            with open("imagenet_labels.json","r") as f:
                classes = json.load(f)
        except:
            # fallback dummy names
            classes = [str(i) for i in range(1000)]
        return model, classes

def preprocess_image(pil_img, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(pil_img).unsqueeze(0)

def apply_colormap_on_image(org_img: np.ndarray, activation: np.ndarray, colormap_name='jet'):
    import matplotlib.pyplot as plt
    # activation is 2D [H, W] normalized 0-1
    heatmap = plt.get_cmap(colormap_name)(activation)[:, :, :3]  # HxWx3 RGBA->RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize((org_img.shape[1], org_img.shape[0]))
    heatmap = np.array(heatmap)
    overlay = (0.5 * heatmap + 0.5 * org_img).astype(np.uint8)
    return overlay

def predict_and_cam(model, classes, pil_img):
    device = torch.device("cpu")
    input_tensor = preprocess_image(pil_img).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        topk = probs.argsort()[-3:][::-1]
    # Grad-CAM needs a backward pass; compute with GradCAM helper
    gradcam = GradCAM(model, target_layer_name="features")
    cam_np, class_idx = gradcam(input_tensor, class_idx=int(topk[0]))
    # Prepare overlay
    org = np.array(pil_img.convert("RGB"))
    cam_resized = Image.fromarray((cam_np*255).astype(np.uint8)).resize((org.shape[1], org.shape[0]))
    cam_arr = np.array(cam_resized) / 255.0
    overlay = apply_colormap_on_image(org, cam_arr, colormap_name='jet')
    # Plain English explanation (simple)
    explanation = f"The model predicted **{classes[class_idx]}** with {probs[class_idx]*100:.1f}% confidence. The highlighted regions (red/yellow) show areas that influenced the decision."
    # Convert top-3 into a dict for Gradio Label component
    top3_dict = { classes[i]: float(probs[i]) for i in topk }
    return classes[class_idx], top3_dict, overlay, explanation

#Gradio UI
def launch_app(model_path=None):
    model, classes = load_model(model_path)
    def infer(image):
        if image is None:
            return "No image", {}, None, "Upload an image."
        pred, top3, overlay, expl = predict_and_cam(model, classes, image)
        
        overlay_pil = Image.fromarray(overlay)
        return pred, top3, overlay_pil, expl

    title = "Tiny Explainable Image Classifier (MobileNetV2 + Grad-CAM)"
    desc = "Upload an image. Model predicts a class and shows a Grad-CAM heatmap overlay (which regions mattered)."
    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}\n\n{desc}")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload image")
            out_col = gr.Column()
            pred_txt = gr.Textbox(label="Predicted class")
            probs = gr.Label(num_top_classes=3, label="Top 3 probabilities")
            cam_img = gr.Image(type="pil", label="Grad-CAM overlay")
            expl_box = gr.Markdown("", label="Explanation")
        btn = gr.Button("Predict")
        btn.click(fn=infer, inputs=[inp], outputs=[pred_txt, probs, cam_img, expl_box])
    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    args = parser.parse_args()
    launch_app(args.model_path)
