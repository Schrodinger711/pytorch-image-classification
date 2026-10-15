# Fashion Image Classification with PyTorch

A computer vision project exploring image classification using PyTorch, from a custom CNN trained on Fashion-MNIST to transfer learning and fine-tuning with ResNet18 on real-world fashion product images.

## Projects

### 1. Fashion-MNIST Classification

A custom Convolutional Neural Network (CNN) that classifies grayscale clothing images into 10 categories.

**Model architecture**
- Two convolutional layers
- ReLU activation
- Two max-pooling operations
- Flatten layer and two fully connected layers
- Cross-entropy loss and Stochastic Gradient Descent (SGD)

**Recorded results**
- Dataset: Fashion-MNIST
- Image size: 28 × 28 pixels, grayscale
- Number of classes: 10
- Batch size: 64
- Learning rate: 0.01
- Training epochs: 15
- Recorded test accuracy: 88.41%

### 2. Fashion Product Classification

A six-class image-classification pipeline built using a pretrained ResNet18 model. The project compares a custom CNN baseline, transfer learning, and fine-tuning.

**Product categories**
- T-shirts
- Shirts
- Casual shoes
- Sports shoes
- Handbags
- Sandals

**Dataset and training**
- 17,818 usable product images
- Training images: 12,470
- Validation images: 2,673
- Test images: 2,675
- Input resolution: 224 × 224 pixels
- Framework: PyTorch and Torchvision

**Model comparison**

| Model | Test accuracy |
|---|---:|
| Custom CNN baseline | 87.66% |
| ResNet18 feature extraction | 92.07% |
| Fine-tuned ResNet18 | 95.63% |

The fine-tuned ResNet18 achieved the highest recorded test accuracy. The test set was used for this comparison, so it should not be treated as an untouched benchmark for future model selection.

**Fine-tuned ResNet18 per-class test accuracy**

| Category | Accuracy |
|---|---:|
| T-shirts | 99.06% |
| Shirts | 98.76% |
| Casual shoes | 86.65% |
| Sports shoes | 88.89% |
| Handbags | 99.62% |
| Sandals | 93.33% |

The results show differences in performance across categories, particularly between shoes and other product types.

## Interactive application

The Streamlit application allows users to upload a product image and view the predicted category and class confidence scores.

**Application features**
- Image upload and preview
- Six-class product prediction
- Confidence scores for all categories
- GPU inference when CUDA is available, with CPU fallback

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Repository contents

- `app.py` — original Fashion-MNIST Streamlit application
- `model.py`, `train.py`, `predict.py` — original CNN workflow
- `product_app.py` — fashion product classification application
- `product_model.py`, `product_dataset.py` — product model and dataset utilities
- `train_baseline.py` — custom CNN training
- `train_resnet.py` — ResNet18 feature extraction
- `train_resnet_finetune.py` — ResNet18 fine-tuning
- `evaluate_baseline.py`, `evaluate_resnet.py`, `evaluate_resnet_finetuned.py` — model evaluation scripts
- `predict_resnet.py` — product image inference
- `prepare_data.py`, `split_data.py`, `explore_data.py` — data preparation and exploration

## Run locally

### 1. Install dependencies

Create and activate a Python virtual environment, then install the required packages.

```bash
python -m venv venv
```

Windows Command Prompt:

```cmd
venv\Scripts\activate
pip install torch torchvision numpy pandas matplotlib streamlit pillow
```

### 2. Run the Fashion-MNIST application

```cmd
python -m streamlit run app.py
```

### 3. Run the product-classification application

```cmd
python -m streamlit run product_app.py
```

The product application requires the trained checkpoint `best_resnet_finetuned.pth` in the project directory. Model checkpoints and datasets are excluded from this repository, so the checkpoint and dataset must be obtained separately before running the corresponding workflows.

## Limitations

- Fashion-MNIST contains small grayscale images and differs substantially from real-world product photographs.
- The product classifier was trained on six selected categories; it is not a general-purpose fashion recognition model.
- Results on images from the internet may differ from test-set results because of backgrounds, image quality, product presentation, and dataset differences.
- The model may be overconfident when predicting unfamiliar images.
- Test accuracy alone does not establish reliable performance in a real-world application.

## Future improvements

- Improve generalization to images from different sources.
- Investigate errors in the shoe categories.
- Add more product categories.
- Evaluate on a separate external dataset.
- Explore deployment and inference optimization.

## Author's note

This project represents a progression from learning CNN fundamentals to applying transfer learning and fine-tuning to a more realistic computer vision task.
