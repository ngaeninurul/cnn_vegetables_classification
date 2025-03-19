# CNN Vegetables Classification

## 📌 Overview
This project develops a Convolutional Neural Network (CNN) model to classify images of various types of vegetables. The model is trained using the **Vegetable Image Dataset** from Kaggle, which consists of 15 different vegetable classes.

## 📂 File Structure
```
cnn_vegetables_classification
├───tfjs_model                 # TFJS is a format for TensorFlow.js, allowing models to run in browsers and JavaScript applications
|   ├───group1-shard1of1.bin
|   └───model.json
├───tflite                     # TF-Lite is an optimized format for mobile and embedded devices
|   ├───model.tflite
|   └───label.txt
├───saved_model                # SavedModel is a standard TensorFlow format for deployment on servers or cloud
|   ├───saved_model.pb
|   └───variables
├───notebook.ipynb             # Main Jupyter notebook for model training
├───README.md                  # Project documentation
├───requirements.txt           # Required dependencies
```

## 📊 Dataset
The dataset used in this project:
[Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Total Images:** 21,000
- **Number of Classes:** 15
- **Data Split:**
  - **Train:** 15,000 images
  - **Validation:** 3,000 images
  - **Test:** 3,000 images

## 🔧 Installation & Usage
1. **Clone Repository:**
   ```bash
   git clone https://github.com/ngaeninurul/cnn_vegetables_classification.git
   cd cnn_vegetables_classification
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
   - Extract the dataset into the `dataset/` folder
4. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `notebook.ipynb` to view and execute the model.

## 🏗 Model Architecture
- **Convolutional Layers:**
  - 2 Conv2D layers (32 & 64 filters, 3x3 kernel, ReLU activation)
  - MaxPooling2D (2x2)
- **Fully Connected Layers:**
  - Flatten layer
  - Dense 128 neurons with ReLU
  - Dropout 0.25
  - Dense 128 neurons with ReLU
  - Dense output 15 neurons with softmax

## 🎯 Training Results
The model was trained for **29 epochs**, achieving **92.97% accuracy** on validation data with a loss of **0.2203**.

