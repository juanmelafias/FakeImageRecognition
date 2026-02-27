# AI-Generated Image Detection

A deep learning project to detect whether an image is real or AI-generated using the CIFAKE dataset.

## Project Structure
```
ImageRecognition/
├── data/
│   ├── raw/           # Downloaded dataset
│   └── processed/     # Preprocessed data
├── models/            # Saved model checkpoints
├── src/
│   ├── config.py      # Configuration and hyperparameters
│   ├── dataset.py     # Data loading and preprocessing
│   ├── model.py       # Model architecture
│   ├── train.py       # Training script
│   ├── evaluate.py    # Model evaluation
│   └── predict.py     # Inference on new images
├── notebooks/         # Jupyter notebooks for exploration
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Download CIFAKE Dataset

**Step 2.1: Set up Kaggle credentials**

Create a `.env` file in the project root with your Kaggle credentials:
```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

To get your Kaggle API credentials:
1. Go to https://www.kaggle.com/settings
2. Scroll to the "API" section
3. Click "Create New API Token"
4. Copy your username and key to the `.env` file

**Step 2.2: Download the dataset**
```bash
python src/download_data.py
```

This will download ~3GB of data (120,000 images) to the `data/raw/` folder. The download may take 5-10 minutes depending on your internet speed.

### 3. Train the Model

```bash
python src/train.py
```

**Training details:**
- **Duration**: 1-2 hours on M4 Pro (varies based on convergence)
- **Output**: Training progress with loss/accuracy per epoch
- **Logs**: Saved to `logs/training_YYYYMMDD_HHMMSS.log`
- **Best model**: Automatically saved to `models/best_model.pth`
- **Expected accuracy**: ~96-97% on validation set

**What to expect during training:**
```
Epoch 1/50 Summary:
  Train Loss: 0.1951 | Train Acc: 92.22%
  Val Loss:   0.1900 | Val Acc:   92.33%
  LR: 0.001000
💾 Saved best model with val_acc: 92.33%
```

Training will automatically stop early if the model stops improving (after 10 epochs without improvement).

### 4. Explore the Dataset (Optional)

Launch the interactive web app to browse the dataset and test the model:
```bash
streamlit run src/app.py
```

This opens a browser with:
- **AI Detector**: Upload images to classify as real or AI-generated
- **Dataset Explorer**: Browse training images with statistics and visualizations

### 5. Evaluate the Model

```bash
python src/evaluate.py
```

This generates:
- Confusion matrix
- ROC curve
- Precision/Recall/F1 scores per class
- Overall test accuracy (~96.8%)
- Results saved to `results/` folder

### 6. Predict on New Images

**Single image:**
```bash
python src/predict.py --image_path path/to/image.jpg
```

**Batch prediction:**
```bash
python src/predict.py --image_folder path/to/folder --output_csv results.csv
```

## Model Performance

**Test Set Results:**
- **Accuracy**: 96.77%
- **Sensitivity (Detecting Fakes)**: 97.03%
- **Specificity (Detecting Real)**: 96.51%
- **ROC-AUC**: ~0.99

**Confusion Matrix:**
```
              Predicted
            REAL    FAKE
Actual REAL 9651    349
      FAKE  297    9703
```

## Dataset Info
- **Name**: CIFAKE (Real and AI-Generated Synthetic Images)
- **Size**: 120,000 images (60,000 real + 60,000 fake)
- **Resolution**: 32×32 pixels
- **Real Images**: From CIFAR-10 dataset
- **Fake Images**: Generated using Stable Diffusion
- **Split**: 80% train, 10% validation, 10% test

## Model Architecture
- Base: ResNet18 (pretrained on ImageNet)
- Modified for binary classification (Real vs AI-generated)
- Optimized for M4 Pro local training

## Hardware Requirements
- RAM: 24GB (recommended)
- Processor: M4 Pro or equivalent
- Training time: ~2-3 hours for 50 epochs
