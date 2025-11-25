# 📩 Ham–Spam Message Classification using Machine Learning (NLP Project)

This project builds a machine learning model to classify text messages as **Ham (Not Spam)** or **Spam** using Natural Language Processing techniques. The project includes data preprocessing, text cleaning, feature extraction (Bag of Words), model training, evaluation, and prediction.

---

## 📁 Project Structure
```
ML_21_Ham_Spam_Project/
│── dataset/
│   └── spam_ham.txt
│── Ham_Spam_Project.ipynb
│── README.md
```

---

## 📊 Dataset Description

Dataset contains two columns:

| Column | Description |
|--------|-------------|
| Target | ham or spam label |
| Msg    | text message |

Dataset Source: Custom dataset (`spam_ham.txt`) with tab-separated values.

---

## 🧹 Data Preprocessing Steps

✔ Loaded and inspected dataset  
✔ Checked target distribution  
✔ Converted text into lowercase  
✔ Removed stopwords  
✔ Applied Bag of Words model using **CountVectorizer**  
✔ Split data into features (X) and labels (y)

---

## 🔠 NLP Feature Extraction

Used **CountVectorizer**:

```python
cv = CountVectorizer(lowercase=True, stop_words='english')
X = cv.fit_transform(corpus)
```

This converts text messages into numerical vectors for training the ML model.

---

## 🤖 ML Models Used

Models applied (as per notebook):

- **Logistic Regression**
- **SGD Classifier**

Pipeline:

```
1. Load dataset
2. Text preprocessing
3. Convert text → vectors (BOW)
4. Train ML model
5. Predict & evaluate
```

---

## 📈 Model Evaluation

Metrics used:

- Accuracy  
- Confusion Matrix  
- Classification Report  

Example (replace with actual values):

```
Accuracy: 97%
Precision: 96%
Recall: 95%
```

---

## 🧠 Prediction Example

```python
sample = ["Congratulations! You've won a prize"]
vector = cv.transform(sample)
model.predict(vector)
```

Output:

```
['spam']
```

---

## ▶️ Run the Project

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/spam-ham-classifier.git
```

### 2️⃣ Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3️⃣ Open Jupyter Notebook
```bash
jupyter notebook ML_21_Ham_Spam_Project.ipynb
```

### 4️⃣ Run all cells

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLP (CountVectorizer)  
- Jupyter Notebook  

---

## 🚀 Future Improvements

- Use TF-IDF Vectorizer  
- Train with Naive Bayes, SVM, RandomForest  
- Deep learning models (LSTM, BERT)  
- Deploy using Streamlit  

---

## 📜 License
Open-source for education & research.

---

## 🙌 Acknowledgements
Dataset prepared for academic Machine Learning practice.
