# 🌟 Fashion Image Classification Using PyTorch

### 🧠 From CNN Fundamentals to Transfer Learning and Fine-Tuning

A computer vision project exploring deep learning for fashion image classification. This project progresses from a custom Convolutional Neural Network (CNN) trained on Fashion-MNIST to a fine-tuned ResNet18 model that classifies real-world fashion product images into six categories.

🌐 **Live Demo:** [Try the Fashion Product Classifier](https://pytorch-image-classification-p6n6aepp9turyyhdxpggh7.streamlit.app/)

💻 **Source Code:** [GitHub Repository](https://github.com/Schrodinger711/pytorch-image-classification)

---

## 🚀 Project Highlights

- 🖼️ Built image classification pipelines using PyTorch and Torchvision.
- 🧠 Implemented a custom CNN for Fashion-MNIST classification.
- 🔄 Applied transfer learning using a pretrained ResNet18 model.
- ⚙️ Fine-tuned ResNet18 to improve product classification performance.
- 📊 Compared three models using validation and test accuracy.
- 🌐 Developed an interactive Streamlit application for image prediction.
- 🎯 Achieved a recorded **95.63% test accuracy** with the fine-tuned ResNet18 model on the selected product dataset.

## 🧩 Projects

### 1️⃣ Fashion-MNIST Classification

A custom CNN that classifies grayscale clothing images into 10 categories.

**🛠️ Model Architecture**

- Two convolutional layers
- ReLU activation
- Two max-pooling operations
- Flatten layer
- Two fully connected layers
- Cross-entropy loss
- Stochastic Gradient Descent (SGD)

**📋 Training Configuration**

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Image dimensions | 28 × 28 pixels |
| Image type | Grayscale |
| Number of classes | 10 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Training epochs | 15 |
| Recorded test accuracy | 88.41% |

### 2️⃣ Fashion Product Classification

A six-class image classification pipeline using a custom CNN baseline, ResNet18 feature extraction, and ResNet18 fine-tuning.

**🛍️ Product Categories**

- 👕 T-shirts
- 👔 Shirts
- 👟 Casual shoes
- 🏃 Sports shoes
- 👜 Handbags
- 🩴 Sandals

**📦 Dataset Overview**

| Split | Number of images |
|---|---:|
| Training | 12,470 |
| Validation | 2,673 |
| Testing | 2,675 |
| **Total** | **17,818** |

The dataset was filtered to include the six selected product categories.

**🧪 Model Comparison**

| Model | Test accuracy |
|---|---:|
| Custom CNN baseline | 87.66% |
| ResNet18 feature extraction | 92.07% |
| Fine-tuned ResNet18 | **95.63%** |

The fine-tuned ResNet18 achieved the highest recorded test accuracy among these three experiments.

**📊 Fine-Tuned ResNet18: Per-Class Test Accuracy**

| Product category | Accuracy |
|---|---:|
| 👕 T-shirts | 99.06% |
| 👔 Shirts | 98.76% |
| 👟 Casual shoes | 86.65% |
| 🏃 Sports shoes | 88.89% |
| 👜 Handbags | 99.62% |
| 🩴 Sandals | 93.33% |

The results show stronger performance on handbags, T-shirts, and shirts, while the two shoe categories remain more challenging.

> 📌 **Evaluation note:** The test set was used to compare the recorded models. The reported test accuracy should therefore be interpreted as an experimental result, not as an untouched benchmark for future model selection.

---

## 🌐 Interactive Web Application

Try the deployed application:

### 👉 [Launch Fashion Product Classifier](https://pytorch-image-classification-p6n6aepp9turyyhdxpggh7.streamlit.app/)

**✨ Features**

- 📤 Upload a product image.
- 🖼️ Preview the uploaded image.
- 🔍 Predict one of six product categories.
- 📈 View confidence scores for all categories.
- 💻 Run inference using the available CPU or CUDA device.

The application provides an interactive demonstration of the trained product classification model.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming language |
| 🔥 PyTorch | Model development and training |
| 👁️ Torchvision | Image transformations and pretrained ResNet18 |
| 📊 NumPy | Numerical operations |
| 🗂️ Pandas | Dataset and CSV processing |
| 📉 Matplotlib | Data visualization |
| 🌐 Streamlit | Interactive web application |
| 🐙 Git and GitHub | Version control and source hosting |

## 📁 Repository Structure

```text
pytorch-image-classification/
│
├── app.py
├── model.py
├── train.py
├── predict.py
│
├── product_app.py
├── product_model.py
├── product_dataset.py
│
├── train_baseline.py
├── train_resnet.py
├── train_resnet_finetune.py
│
├── evaluate_baseline.py
├── evaluate_resnet.py
├── evaluate_resnet_finetuned.py
│
├── predict_resnet.py
├── prepare_data.py
├── split_data.py
├── explore_data.py
│
├── tensor_basics.py
├── requirements.txt
├── .gitignore
└── README.md
```

📌 Local datasets, generated CSV files, virtual environments, and trained model checkpoints are excluded from the repository.

## ⚙️ Run the Project Locally

### 1. 📥 Clone the repository

```bash
git clone https://github.com/Schrodinger711/pytorch-image-classification.git
cd pytorch-image-classification
```

### 2. 🐍 Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows Command Prompt:

```cmd
venv\Scripts\activate
```

### 3. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### 4. 🌐 Launch the product classifier

```bash
python -m streamlit run product_app.py
```

**⚠️ Required files:** The product classifier requires the trained checkpoint `best_resnet_finetuned.pth` in the project directory. The dataset and checkpoint are not included in this repository, so obtain the required files separately before running the app or training pipeline. Dataset CSV files and paths may also need to be generated or configured for your local environment.

To run the original Fashion-MNIST application instead:

```bash
python -m streamlit run app.py
```

## ⚠️ Limitations

- 🖼️ Fashion-MNIST contains small grayscale images and differs from real-world product photographs.
- 🛍️ The product classifier supports only six selected categories.
- 🌍 Predictions on external images may be less accurate because of backgrounds, lighting, image quality, and differences between datasets.
- 👟 The shoe categories are more challenging than several other classes in the recorded evaluation.
- 📊 Test-set performance does not guarantee the same accuracy on images uploaded by users.
- 🎯 Model confidence scores should not be interpreted as guarantees that predictions are correct.

## 🔮 Future Improvements

- 🧪 Evaluate the model on a separate external dataset.
- 👟 Investigate classification errors between casual and sports shoes.
- 🛍️ Expand the number of supported product categories.
- 📱 Improve generalization across image sources and backgrounds.
- ⚡ Explore inference optimization and deployment improvements.
- 📈 Add confusion matrices and additional evaluation visualizations.

## 👨‍💻 Project Goal

This project represents a progression from learning CNN fundamentals to applying transfer learning, fine-tuning, model evaluation, and web deployment to a practical computer vision problem.

The broader goal is to build a stronger foundation in AI/ML and develop useful, real-world applications.

---

⭐ **If you find this project interesting, explore the source code and try the live demo!**