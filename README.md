# SMS Spam Detection using NLP
**DLBAIPNLP01 – Project: NLP - Task 2**

Author: Martin Lana Bengut  
Matriculation Number: 92125626  
Tutor: Simon Martin  
Date: November 2025

---

## 🎯 Project Overview

Natural Language Processing system for automatic SMS spam classification using machine learning. Implements and compares multiple algorithms with progressive evaluation on datasets of increasing size.

## 📊 Results

**Final Performance (5,572 messages):**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.11% | 99.16% | 79.19% | 88.06% |
| **SVM** | **98.56%** | 97.84% | 91.28% | 94.44% |

**Progressive Evaluation:**
- Small (1,000): SVM 95.5%
- Medium (3,000): SVM 98.0%
- Large (5,572): SVM 98.56%

✅ Exceeded 95% accuracy target

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/martinlanabengut/spam-detection-nlp.git
cd spam-detection-nlp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Run Analysis

```bash
python src/complete_analysis.py
```

This will:
1. Load SMS Spam Collection dataset
2. Preprocess text (cleaning, tokenization, stemming)
3. Train models progressively on 1K, 3K, and full dataset
4. Generate all visualizations and results
5. Save trained models

## 📁 Project Structure

```
NLP Spam Detection Project/
├── data/
│   └── SMSSpamCollection          # Dataset (5,572 messages)
├── src/
│   ├── spam_detector.py           # Main detector class
│   └── complete_analysis.py       # Progressive evaluation pipeline
├── outputs/
│   ├── results_table.csv          # Performance metrics
│   └── results_summary.json       # Detailed results
├── visualizations/
│   ├── confusion_matrix_*.png     # 6 confusion matrices
│   ├── performance_by_size.png    # Comparison chart
│   └── roc_curves.png             # ROC analysis
├── models/
│   ├── spam_detector_nb.pkl       # Trained Naive Bayes
│   └── spam_detector_svm.pkl      # Trained SVM
├── PROJECT_REPORT_FINAL.tex       # LaTeX report
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔬 Methodology

### Preprocessing Pipeline
1. Text normalization (lowercase, URL/email removal)
2. Tokenization (word splitting)
3. Stopword removal (179 English stopwords)
4. Stemming (Porter Stemmer)
5. TF-IDF vectorization (3,000 features, bigrams)

### Models Implemented
- **Naive Bayes:** Probabilistic baseline, fast training
- **SVM:** Linear kernel, optimal hyperplane separation

### Evaluation Strategy
- Progressive datasets: 1K → 3K → 5.5K samples
- 80/20 train-test split with stratification
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## 📈 Key Findings

1. **SVM outperforms Naive Bayes** consistently (1-5% accuracy advantage)
2. **Performance improves with data size** but plateaus after 3K samples
3. **Recall most sensitive to dataset size** (27% → 91% for SVM)
4. **High ROC-AUC (>98%)** indicates excellent discriminative ability
5. **Low false positive rate (1.2%)** suitable for production deployment

## 🛠️ Technologies

- **Python 3.10+**
- **NLTK** - Text processing
- **scikit-learn** - ML algorithms
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization

## 📊 Dataset

**SMS Spam Collection**
- Source: UCI ML Repository
- Size: 5,572 SMS messages
- Classes: Spam (13.4%), Ham (86.6%)
- Language: English

## 📝 Documentation

Complete project report available in `PROJECT_REPORT_FINAL.tex`

Compile with:
- Overleaf (recommended): https://www.overleaf.com/
- Or local: `pdflatex PROJECT_REPORT_FINAL.tex`

## 🔗 References

- Jurafsky, D. & Martin, J. (2013). Speech and language processing
- Almeida et al. (2011). SMS spam filtering: New collection and results
- UCI ML Repository: SMS Spam Collection

## 📧 Contact

Martin Lana Bengut  
IU Internationale Hochschule  
DLBAIPNLP01 Project


