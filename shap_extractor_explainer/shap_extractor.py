# Script 1: shap_extractor.py
"""
Takes as input a directory of videos or a single video file.
Extracts SHAP values per key frame using a ViT model.
Saves shap values for each video as a .npz file in an output directory.
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import shap
from torchvision.models import vit_b_16

def extract_frames(video_path, interval=30):
    """Yield frames at every `interval` frames."""
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, Image.fromarray(img)
        idx += 1
    cap.release()


def compute_shap_for_video(video_path, model, preprocess, explainer, device):
    """Compute SHAP values for sampled frames of a video."""
    shap_vals = []
    indices = []
    for idx, frame in extract_frames(video_path):
        inp = preprocess(frame).unsqueeze(0).to(device)
        shap_value = explainer.shap_values(inp)
        shap_vals.append(np.array(shap_value))
        indices.append(idx)
    return indices, np.stack(shap_vals)


def main(input_path, output_dir, interval=30, gpu=True):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    model = vit_b_16(pretrained=True).to(device)
    model.eval()

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    background = torch.zeros((1, 3, 224, 224), device=device)
    explainer = shap.GradientExplainer(model, background)

    paths = []
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                paths.append(os.path.join(input_path, fname))
    else:
        paths = [input_path]

    for vid in paths:
        print(f"Processing {vid}...")
        idxs, vals = compute_shap_for_video(vid, model, preprocess, explainer, device)
        out_name = os.path.splitext(os.path.basename(vid))[0] + '_shap.npz'
        np.savez_compressed(os.path.join(output_dir, out_name), indices=idxs, shap=vals)
        print(f"Saved SHAP values to {out_name}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Video file or directory')
    parser.add_argument('--output', required=True, help='Output directory for shap files')
    parser.add_argument('--interval', type=int, default=30, help='Frame sampling interval')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Disable GPU')
    args = parser.parse_args()
    main(args.input, args.output, args.interval, args.gpu)

# example usage
"""python shap_extractor.py \
  --input /path/to/your/videos_dir \
  --output /path/to/shap_output_dir \
  --interval 30
"""
