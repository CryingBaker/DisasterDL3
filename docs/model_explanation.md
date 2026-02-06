
# Model Architecture & Usage Guide

## 1. How It Works (Simplified)
This model is designed to detect floods by comparing "Before" and "After" radar images.

*   **Input 1: Post-Event SAR (Radar):** Shows the ground situation *during* the flood. Water looks dark.
*   **Input 2: Pre-Event SAR (Radar):** Shows the normal ground situation.
*   **Input 3: Infrastructure (OSM):** A map of roads and buildings to tell the model "this is a city" or "this is a highway".

### The "Siamese" Logic
The model uses a **Siamese Network** structure. This means it uses the *same* pair of "eyes" (Encoder) to look at both the Before and After images. It extracts features from both and then calculates the **difference**.

If a pixel changes from "Bright" (Land) to "Dark" (Water), the model flags it as **FLOOD**.

## 2. Technical Architecture
*   **Backbone:** ResNet-18 (Pre-trained on ImageNet).
*   **Fusion Strategy:** Concatenation of [Post, Pre, Difference] features.
*   **Infrastructure Injection:** Road/Building masks are encoded separately and injected into the decoder to guide the segmentation boundaries (e.g., floods often stop at raised highways).
*   **Output:** A probability map (0-1) where 1 = Flood.

## 3. Training & Data
*   **Loss Function:** Binary Cross Entropy (BCE).
*   **Metrics:** IoU (Intersection over Union) - the gold standard for segmentation.
*   **Data Handling:** Automatically handles missing Pre-Event data by zero-filling (fallback mode).

## 4. Web Integration Flow
1.  **User Selects Area:** The web frontend sends a bounding box.
2.  **Backend Fetches Data:** Python scripts download Sentinel-1 and OSM data for that area.
3.  **Preprocessing:** Data is resized to 512x512.
4.  **Inference:** The model predicts the flood mask.
5.  **Overlay:** The result is displayed as a Blue layer on the map.
