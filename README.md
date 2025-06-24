# Histopathologic Cancer Detection - Deep Learning Project

A comprehensive deep learning solution for binary image classification in the Kaggle "Histopathologic Cancer Detection" competition using TensorFlow/Keras.

## Project Overview

This project implements both custom CNN architectures and transfer learning approaches to detect metastatic cancer in histopathologic scan images (96x96 pixels). The solution includes comprehensive data analysis, preprocessing, model training, evaluation, and submission preparation.

## Dataset

- **Source**: Kaggle - Histopathologic Cancer Detection
- **Image Size**: 96x96 pixels (RGB)
- **Task**: Binary classification (cancer vs no cancer)
- **Labels**: Stored in `train_labels.csv`
- **Evaluation Metric**: Area Under the ROC Curve (AUC)

## Directory Structure

```
CNN-Cancer-Detection/
├── histopathologic_cancer_detection.ipynb    # Main notebook
├── requirements.txt                          # Python dependencies
├── README.md                                # This file
├── data/                                    # Data directory (not included)
│   ├── train/                              # Training images
│   ├── test/                               # Test images
│   └── train_labels.csv                    # Training labels
├── checkpoints/                            # Model checkpoints (created during training)
├── models/                                 # Saved models (created during training)
└── submissions/                            # Generated submission files
```

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
   - Go to: https://www.kaggle.com/c/histopathologic-cancer-detection
   - Download and extract the dataset to the `data/` directory

## Usage

1. **Open the Jupyter Notebook**:
```bash
jupyter notebook histopathologic_cancer_detection.ipynb
```

2. **Update Data Paths**: Modify the `BASE_PATH` variable in the notebook to point to your dataset location.

3. **Run the Notebook**: Execute cells sequentially to:
   - Load and explore the data
   - Perform exploratory data analysis
   - Preprocess images with augmentation
   - Train the model (custom CNN or transfer learning)
   - Evaluate model performance
   - Generate test predictions
   - Save the trained model

## Key Features

### Data Analysis
- Class distribution analysis
- Sample image visualization
- Pixel value distribution analysis

### Preprocessing
- Image normalization (rescaling to [0,1])
- Data augmentation (rotation, shifts, shear, zoom, flips)
- Stratified train-validation split

### Model Architectures
1. **Custom CNN**: Multi-layer convolutional network with batch normalization and dropout
2. **Transfer Learning**: Pre-trained MobileNetV2 or EfficientNetB0 with custom classifier head

### Training Features
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Learning rate reduction on plateau
- Class weight balancing for imbalanced data

### Evaluation
- Comprehensive metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC)
- Confusion matrix visualization
- ROC curve plotting
- Prediction distribution analysis

### Output Files
- Trained model (`./models/model_name_final.h5`)
- Model summary and info files
- Training history
- Kaggle submission file (`submission_model_name.csv`)

## Model Configuration

The notebook allows easy switching between model types by changing the `MODEL_CHOICE` variable:
- `'custom'`: Use custom CNN architecture
- `'transfer'`: Use transfer learning with MobileNetV2

## Performance Tips

1. **GPU Usage**: Ensure TensorFlow can access GPU for faster training
2. **Batch Size**: Adjust `BATCH_SIZE` based on available memory
3. **Learning Rate**: Fine-tune learning rates for different architectures
4. **Augmentation**: Experiment with different augmentation parameters

## Customization

The notebook is designed to be easily customizable:
- **Data Paths**: Update paths to match your dataset location
- **Model Architecture**: Modify the model creation functions
- **Hyperparameters**: Adjust training parameters in dedicated sections
- **Augmentation**: Customize data augmentation strategies

## Results

The project generates:
- Detailed training history plots
- Comprehensive evaluation metrics
- Visual analysis of model performance
- Ready-to-submit prediction file

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- GPU recommended for faster training
- At least 8GB RAM for comfortable execution

## Troubleshooting

1. **Out of Memory**: Reduce batch size or image resolution
2. **Slow Training**: Ensure GPU is being used, reduce model complexity
3. **Poor Performance**: Increase training epochs, adjust learning rate, or try different architectures

## Contributing

Feel free to fork this project and submit improvements. Some areas for enhancement:
- Additional model architectures
- Advanced augmentation techniques
- Ensemble methods
- Model interpretability features

## License

This project is for educational purposes. Please respect Kaggle's terms of service when using competition data.