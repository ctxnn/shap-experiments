# AI vs Fake Media Detection with SHAP Explanations

This repository contains experiments for detecting AI-generated (fake) images and videos using deep learning models and explaining their predictions with SHAP (SHapley Additive exPlanations).

## Project Overview

The project focuses on:
1.  **Training a Deep Learning Model:** A ResNet-18 model is trained to classify images as either real or AI-generated.
2.  **Explainability with SHAP:** SHAP is used to understand which parts of an image contribute most to the model's decision. This helps in identifying visual artifacts or patterns that the model uses to detect fakes.
3.  **Video Analysis:** The image-based detection approach is extended to videos by processing individual frames.
4.  **VLM-Enhanced Descriptions:** A Vision Language Model (VLM) like BLIP is used to generate textual descriptions of why an image is classified as fake, incorporating insights from SHAP/feature importance maps.

## Notebooks

*   **`aivsfake-shap.ipynb`**:
    *   Loads and preprocesses a dataset of real and fake face images.
    *   Defines and trains a `FakeImageDetector` model (ResNet-18 based).
    *   Performs SHAP analysis on the trained model to generate explanations for its predictions.
    *   Generates textual descriptions based on SHAP values, highlighting regions and potential artifacts.
    *   Visualizes the original images alongside their SHAP attribution maps and descriptions.
*   **`aivsfake-shap-with-video.ipynb`**:
    *   Similar to `aivsfake-shap.ipynb` but includes an enhanced `VLMDescriptionGenerator`.
    *   This generator uses a gradient-based feature importance method (as an alternative to SHAP for some cases) and integrates a VLM (Salesforce/blip-image-captioning-base) to produce more descriptive explanations.
    *   It processes images, saves processed versions, heatmaps, and overlays.
    *   Generates an HTML report and an Excel file summarizing the detection results, including images, heatmaps, VLM-generated descriptions, and confidence scores.
*   **`video-stuff/deepfake-video-detection.ipynb`**:
    *   Focuses on deepfake detection in videos.
    *   Uses a pre-trained Vision Transformer (ViT) model (`Wvolf/ViT_Deepfake_Detection`) and fine-tunes it on a dataset of original and manipulated videos.
    *   Extracts frames from videos, aggregates them, and feeds them to the model.
    *   Implements SHAP analysis (trying both `PartitionExplainer` and `GradientExplainer`) to explain frame-level predictions.
    *   Uses a BLIP model to generate captions for video frames.
    *   Saves visualizations of original frames, SHAP overlays, and generated captions.

## Key Features and Techniques

*   **Deep Learning for Fake Detection:** Utilizes ResNet-18 and Vision Transformer (ViT) architectures.
*   **SHAP (SHapley Additive exPlanations):** Provides model-agnostic explanations for predictions, highlighting important image regions.
*   **Vision Language Models (VLMs):** Employs models like BLIP to generate natural language descriptions based on image content and model insights.
*   **Data Handling:** Includes custom PyTorch `Dataset` and `DataLoader` implementations for images and video frames.
*   **Output Generation:**
    *   Visualizations of SHAP values and feature importance heatmaps.
    *   JSON files with detailed descriptions and prediction results.
    *   CSV and Excel reports for batch processing.
    *   HTML reports for easy viewing of results.

## Output Structure

The `outputs/` directory stores:
*   `best_model.pth`: The trained weights of the fake image detection model.
*   `descriptions.json`: JSON file containing SHAP-based descriptions for analyzed images.
*   `fake_detection_results.csv`: CSV file with results from the `VLMDescriptionGenerator`.
*   `shap_vis_*.png`: SHAP visualization images.
*   `heatmaps/`: Directory containing feature importance heatmaps.
*   `overlays/`: Directory containing images overlaid with heatmaps.
*   `processed_images/`: Directory containing processed versions of input images.

The `video-stuff/img-analysis/` and `video-stuff/vid-analysis/` directories store:
*   SHAP visualizations for individual frames and video sequences.
*   Markdown files with analysis notes.

## How to Run

1.  **Set up your environment:** Ensure you have Python and necessary libraries (PyTorch, torchvision, SHAP, transformers, OpenCV, Matplotlib, scikit-learn, Pillow, tqdm, pandas, openpyxl) installed.
2.  **Prepare your data:** The notebooks expect data in a specific structure (e.g., `/kaggle/input/real-vs-fake-faces/`). You might need to adjust the `DATA_ROOT` in the `Config` class within the notebooks.
3.  **Run the Jupyter notebooks:**
    *   Execute the cells in `aivsfake-shap.ipynb` or `aivsfake-shap-with-video.ipynb` for image-based fake detection and explanation.
    *   Execute the cells in `video-stuff/deepfake-video-detection.ipynb` for video-based deepfake detection and explanation.

## Dependencies

*   Python 3.x
*   PyTorch
*   torchvision
*   SHAP
*   transformers
*   OpenCV (cv2)
*   Matplotlib
*   scikit-learn
*   Pillow (PIL)
*   tqdm
*   pandas
*   openpyxl (for Excel export)

