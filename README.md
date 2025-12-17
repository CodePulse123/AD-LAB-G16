# 📰 Fake News Detection Using Machine Learning

## 📌 Project Overview

Fake news has become a serious issue in the digital era, spreading misinformation rapidly through social media and online platforms. This project aims to **detect fake news articles using Machine Learning techniques** by analyzing textual content and classifying it as **Fake** or **Real**.

The system uses Natural Language Processing (NLP) for text preprocessing and machine learning algorithms for classification.

---

## 🎯 Objectives

* To identify whether a news article is **Fake or Real**
* To apply **NLP techniques** on text data
* To train and evaluate machine learning models for text classification
* To reduce the spread of misinformation

---

## 🛠️ Technologies Used

* **Back end:** Python,Node.js,express.js,APIs
* **Front end:** HTML,CSS,JS,React
* **Libraries:**

  * NumPy
  * Pandas
  * Scikit-learn
  * Matplotlib / Seaborn (for visualization)
* **IDE:** Jupyter Notebook / VS Code

---

## 📂 Dataset

* The dataset contains labeled news articles:

  * **Fake News**
  * **Real News**
* Features include:

  * News title
  * News text
  * Label (Fake / Real)

---

## ⚙️ Methodology

1. **Data Collection** – Load and explore the dataset
2. **Data Preprocessing**

   * Removal of stopwords
   * Tokenization
   * Stemming / Lemmatization
3. **Feature Extraction**

   * TF-IDF Vectorization
   * Bag of Words (BoW)
4. **Model Training**

   * Logistic Regression
   * Naive Bayes
   * Support Vector Machine (SVM)
5. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * Confusion Matrix

---

## 📊 Results

* The trained model successfully classifies news articles with good accuracy.
* Performance metrics indicate effective detection of fake news.

---

## ▶️ How to Run the Project

1. Clone or download the repository
2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script
4. Provide a news article as input to get prediction

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── dataset/
│   └── news.csv
├── notebooks/
│   └── fake_news_detection.ipynb
├── model/
│   └── trained_model.pkl
├── README.md
└── requirements.txt
```

---

## 🚀 Future Enhancements

* Use Deep Learning models (LSTM, BERT)
* Real-time fake news detection
* Web-based interface
* Multi-language support

