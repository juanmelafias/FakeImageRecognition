# Face Recognition with LFW Dataset

Fine-tuning a face recognition model on the **Labeled Faces in the Wild (LFW)** dataset using triplet loss and pre-trained ResNet backbone. Optimized for Mac M4 Pro with MPS acceleration.

## 🎯 Project Overview

This project implements a face recognition system that:
- Uses **triplet loss** for learning face embeddings
- Fine-tunes a **pre-trained ResNet-50** on LFW dataset
- Achieves face verification on the standard LFW pairs benchmark
- Leverages **Apple Silicon MPS** for accelerated training

## 📁 Project Structure

```
FaceRecognition/
├── data/
│   ├── raw/              # LFW dataset (downloaded)
│   └── processed/        # Processed data (if any)
├── models/               # Saved model checkpoints
├── logs/                 # TensorBoard logs
├── results/              # Evaluation plots and metrics
├── notebooks/            # Jupyter notebooks for exploration
└── src/
    ├── config.py         # Configuration and hyperparameters
    ├── download_data.py  # Download LFW dataset
    ├── dataset.py        # Dataset loader with triplet sampling
    ├── model.py          # Face embedding model architecture
    ├── train.py          # Training script
    └── evaluate.py       # Evaluation on LFW pairs
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd FaceRecognition
pip install -r requirements.txt
```

### 2. Download LFW Dataset

```bash
python src/download_data.py
```

This will download:
- **LFW images** (~173 MB, 13,233 images of 5,749 people)
- **pairs.txt** (for face verification evaluation)
- **lfw-names.txt** (list of people in dataset)

### 3. Train the Model

```bash
python src/train.py
```

**Training Configuration:**
- Backbone: ResNet-50 (pre-trained on ImageNet)
- Embedding dimension: 512
- Triplet loss margin: 0.2
- Batch size: 32
- Epochs: 30
- Learning rate: 0.0001
- Device: Apple MPS (Metal Performance Shaders)

**Expected training time on M4 Pro:** ~2-4 hours

### 4. Evaluate the Model

```bash
python src/evaluate.py
```

Evaluation metrics include:
- Accuracy on LFW pairs
- ROC AUC
- False Accept Rate (FAR)
- False Reject Rate (FRR)
- Confusion matrix
- Distance distribution plots

## 📊 Dataset Information

**LFW (Labeled Faces in the Wild)**
- 13,233 face images
- 5,749 unique people
- Images: 250×250 pixels (resized to 160×160 for training)
- Natural, unconstrained face images

**People with ≥10 images:** ~158 people (used for training)
- These provide enough samples for triplet mining
- Remaining people used for validation

## 🧠 Model Architecture

**Face Embedding Model:**
```
Input: RGB image (160×160×3)
↓
ResNet-50 backbone (pre-trained on ImageNet)
↓
Global Average Pooling
↓
Fully Connected Layer (2048 → 512)
↓
Batch Normalization
↓
L2 Normalization
↓
Output: 512-dim embedding
```

**Triplet Loss:**
- Minimizes distance between anchor and positive (same person)
- Maximizes distance between anchor and negative (different person)
- Margin: 0.2

## 💻 Hardware Requirements

**Tested on:**
- Mac M4 Pro (24 cores, 24GB RAM)
- Apple MPS backend for PyTorch

**Alternatives:**
- CUDA GPU (NVIDIA)
- CPU (slower, not recommended)

## 📈 Monitoring Training

Monitor training with TensorBoard:

```bash
tensorboard --logdir=logs/
```

Open http://localhost:6006 in your browser.

## 🎛️ Configuration

Edit `src/config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 160 | Input image resolution |
| `EMBEDDING_DIM` | 512 | Face embedding dimension |
| `MODEL_BACKBONE` | resnet50 | Backbone architecture |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 30 | Number of training epochs |
| `LEARNING_RATE` | 0.0001 | Learning rate |
| `MARGIN` | 0.2 | Triplet loss margin |
| `VERIFICATION_THRESHOLD` | 0.6 | Distance threshold for verification |

## 📝 Usage Example

```python
import torch
from src.model import load_model, verify_faces
from src.dataset import get_transforms
from PIL import Image

# Load trained model
model = load_model('models/best_model.pth')

# Load and preprocess images
_, transform = get_transforms(augment=False)
img1 = transform(Image.open('person1_photo1.jpg')).unsqueeze(0)
img2 = transform(Image.open('person1_photo2.jpg')).unsqueeze(0)

# Verify if same person
result = verify_faces(model, img1, img2, threshold=0.6)
print(f"Same person: {result['is_same_person']}")
print(f"Distance: {result['distance']:.3f}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📚 Key Concepts

**Face Recognition vs Face Verification:**
- **Recognition:** Who is this person? (Identification)
- **Verification:** Are these two faces the same person? (This project)

**Triplet Loss:**
- Learns embeddings where same person faces cluster together
- Different people's faces pushed apart
- More robust than classification loss for face recognition

**Fine-tuning:**
- Start with ImageNet pre-trained ResNet
- Freeze early layers (general features)
- Train embedding layer on face data
- Gradually unfreeze layers if needed

## 🔧 Troubleshooting

**Out of memory errors:**
- Reduce `BATCH_SIZE` in config.py (try 16 or 8)
- Reduce `IMAGE_SIZE` (try 128 instead of 160)

**Slow training:**
- Check MPS is enabled: `torch.backends.mps.is_available()`
- Reduce `NUM_WORKERS` in config.py

**Dataset not downloading:**
- Manual download from: http://vis-www.cs.umass.edu/lfw/
- Extract to `data/raw/`

## 📖 References

- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832) - Triplet loss for face recognition
- [ArcFace Paper](https://arxiv.org/abs/1801.07698) - Advanced face recognition loss

## ⚖️ License & Ethics

**Dataset License:** LFW is for research and educational purposes only.

**Ethical Considerations:**
- Face recognition has privacy implications
- Only use with consent of individuals
- Do not deploy for surveillance without legal approval
- Be aware of bias in face recognition systems

## 🎯 Next Steps

- [ ] Try different backbones (ResNet-18, MobileNet)
- [ ] Implement hard negative mining
- [ ] Add real-time webcam face verification
- [ ] Create Streamlit demo app
- [ ] Fine-tune on custom face dataset
- [ ] Explore other loss functions (ArcFace, CosFace)

## 👨‍💻 Author

Built as part of the ImageRecognition workspace for learning face recognition with deep learning.

---

**Happy Face Recognition! 🤖👤**
