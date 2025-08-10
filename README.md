# 🛍️ Product Category Classification with DistilBERT

This project uses a pre-trained transformer model, DistilBERT, to build a product category classifier for e-commerce data. The model is fine-tuned on a labeled dataset of product descriptions to learn how to assign each item to one of four categories.  The entire workflow was developed and executed in Google Colab to take advantage of free GPU resources. A Tesla T4 GPU was used to speed up training and evaluation. The workflow covers everything from data preparation and model training to evaluation and real-world inference. The goal is to create a fast and reliable solution that can be used to automate product tagging in online retail systems.

## 🗃️ Repository Structure

```plaintext
Product-category-classifier/
│
├── data/                                      # Dataset
│   ├── raw/                                   # Raw data
│   └── processed/                             # Processed data
│
├── figures/                                   # Visualizations
│   ├── correlation-matrix.png        
│   └── product-category-distribution.png  
│
├── models/                                    # Trained models
│
├── notebooks/                                 # Jupyter Notebooks
│   └── product-category-classification.ipynb  # End-to-end project notebook
│
├── results/                                   # Model output
│   └── metrics                                # Model metrics
│       └── model-evaluation-metrics.txt
│   └── predictions                            # Model predictions
│       └── predictions_output.txt                       
│
├── colab_setup.py                             # Colab set up files
├── requirements.txt                           # Required dependencies
└── README.md                                  # Project documentation
```

## 📘 Project Overview

- **Introduction** – Fine-tuned DistilBERT to classify e-commerce product descriptions into Electronics, Household, Books, and Clothing & Accessories.

- **Data Loading and Preparation** – Cleaned dataset by removing missing entries, shuffled data, and mapped labels to numerical IDs.

- **Data Splitting and Tokenization** – Performed stratified train-test split and tokenized product descriptions using DistilBERT tokenizer with padding and truncation.

- **Data Collation** – Applied dynamic padding during batching for efficient training.

- **Model Setup and Fine-Tuning** – Loaded pre-trained DistilBERT for sequence classification and fine-tuned it on the product description dataset.

- **Training Configuration** – Set training parameters including batch size, epochs, weight decay, and evaluation strategy.

- **Evaluation Metrics** – Used accuracy, precision, recall, and F1 score to monitor performance, achieving around 96.5% across metrics.

- **Inference Pipeline** – Created a pipeline for real-time product category predictions with confidence scores.

- **Conclusion** – Delivered an accurate, efficient model ready for automating product categorization in e-commerce applications.

## ⚙️ Dependencies

This project requires the following Python libraries:

```bash
pip install -r requirements.txt
```
- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **Scikit-learn**
- **Pandas**
- **Numpy**

## ▶️ How to Run the Project

### Option 1: Run Locally with GPU

1. Clone this repository:

   ```bash
   git clone https://github.com/herrerovir/Product-category-classifier
   ```

2. Navigate to the project directory:

   ```bash
   cd Product-category-classifier
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook to run the project:

   ```bash
   jupyter notebook
   ```

5. Follow the code to load the dataset, preprocess the data, fine-tune the DistilBERT model, and perform inference on new product descriptions.

### Option 2: Run on Google Colab (Recommended if no GPU locally)

1. Open a new notebook in Google Colab.

2. Clone the repository inside the notebook:

   ```bash
   !git clone https://github.com/herrerovir/Product-category-classifier
   ```

3. Navigate to the cloned folder and open the notebook `Product-category-classification.ipynb`.

4. Set runtime type to GPU and select **Tesla T4**.

5. Run the notebook cells or scripts to execute the project.

*Colab’s Tesla T4 GPU accelerates training and evaluation without any local setup.*

## 📂 Model Files

The trained model files are **not included** in this repository due to their large size. Since the project runs in Google Colab, the fine-tuned model is saved directly to your Google Drive during training. The `colab_setup.py` script in the root directory automatically creates all necessary folders to organize and store the model and related outputs once you run the project.

When you run the notebook in Colab, your trained model will be saved to the corresponding folder in your Drive, making it easy to load for inference or further training without needing to download from this repo.

## 📊 Model Performance

The model delivers consistently strong results across all key metrics, generalizes effectively on new data, and produces confident, reliable predictions.

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 96.51%    |
| Precision  | 96.52%    |
| Recall     | 96.51%    |
| F1 Score   | 96.51%    |
| Eval Loss  | 0.2059    |

## 🚀 Inference Examples

The model confidently classifies product descriptions into their correct categories with high certainty. Here are a few examples showcasing its predictions:

- **Input:** Samsung Galaxy Tab S9 Ultra with 14.6'' AMOLED Display and S Pen. **Prediction:** Electronics (Confidence: 99.88%)

- **Input:** Atomic Habits by James Clear – Build Good Habits & Break Bad Ones. **Prediction:** Books (Confidence: 99.99%)

- **Input:** Levi’s Men's 511 Slim Fit Jeans – Stretch Denim, Dark Indigo. **Prediction:** Clothing & Accessories (Confidence: 99.96%)

These results highlight the model’s ability to accurately understand diverse product descriptions and assign the right category with near-perfect confidence.

## 📈 Results

The fine-tuned DistilBERT model achieved strong performance with over 96% accuracy, precision, recall, and F1 score on the test set. It reliably categorizes product descriptions across Electronics, Household, Books, and Clothing & Accessories. During inference, the model outputs highly confident predictions, making it well-suited for practical e-commerce applications.

## 🙌 Acknowledgments

Built using [Hugging Face Transformers](https://huggingface.co/transformers/), [Datasets](https://huggingface.co/docs/datasets/), and PyTorch.
