# SLUM AREA DETECTION IN DHAKA CITY, BANGLADESH, USING SATELLITE REMOTE SENSING AND DEEP LEARNING

This project uses deep learning (U-Net with Attention Gates) to identify slum areas in Dhaka City using high-resolution satellite imagery (Maxar- 0.3m).

## 🛰️ Data Sources

- **Satellite Imagery (0.3m Maxar)**  
  Open-access from [OpenAerialMap](https://map.openaerialmap.org/)

- **Slum Area Polygons (Ground Truth)**  
  World Bank (2017) [Download Dataset](https://datacatalog.worldbank.org/search/dataset/0041703/Dhaka--Bangladesh----Informal-Settlements--ESA-EO4SD-Urban-)

## 🔍 Methodology

1. **Preprocessing**:
   - Clipped imagery to study the area (Only focused on the known large slum areas)
   - Rasterized slum shapefiles to create labeled masks
   - Split image and mask patches (128x128 pixels) into:
     - 75% Training set
     - 25% Testing set

2. **Model Architecture**:
   - U-Net with integrated Attention Gates
   - Model defined in `models/simple_multi_unet_model_attentiongate.py`
   - Compiled and trained using TensorFlow/Keras

3. **Evaluation**:
   - Pixel-wise accuracy
   - Class-wise F1 scores
   - Visual comparison of predictions vs. ground truth

## 📁 Repository Contents

- `notebooks/UnetAG_maxar_slum_detection.ipynb`: Full notebook from data loading to prediction
- `models/simple_multi_unet_model_attentiongate.py`: U-Net model definition with attention
- `data_samples/`: Sample input image and mask for illustration


## 🚫 Data Note

Due to size constraints, the full dataset is not included in this repository. Only a sample image and mask are included here. All data was accessed from public sources and processed offline. You may request more info if needed.


---



