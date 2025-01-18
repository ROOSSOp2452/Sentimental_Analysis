
# Sentiment Analysis Project

This project performs sentiment analysis on text data to classify the sentiment as positive, negative, or neutral. The model achieves an accuracy of **90%** in predicting the sentiment of text data. The project is implemented using Python and Jupyter Notebook.

## Project Overview

The **Sentiment Analysis** model classifies text data into different sentiment categories. The dataset consists of labeled text data (positive, negative, neutral), which is used to train and evaluate machine learning models. The project uses techniques like data preprocessing, feature extraction, and machine learning to achieve high accuracy in sentiment classification.

## Files in this Repository

- **Sentimental_analysis.ipynb**: Jupyter notebook containing the entire process, including data preprocessing, model building, training, and evaluation.
- **train.csv**: The training dataset containing text samples with their corresponding sentiment labels.
- **test.csv**: The test dataset used for model evaluation.
- **testdata.csv**: Additional test data for evaluating the model.
- **training.csv**: Another training dataset variant used for improving model accuracy.

## Key Features

- **Data Preprocessing**: Text cleaning (removal of stopwords, tokenization, stemming/lemmatization), text vectorization using TF-IDF or word embeddings.
- **Model Training**: Trained on multiple models like Logistic Regression, Naive Bayes, and Support Vector Machine (SVM).
- **Model Evaluation**: Achieved **90% accuracy** in predicting sentiment. Evaluated using performance metrics like accuracy, precision, recall, and F1-score.

## Libraries Used

- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Machine learning algorithms and model evaluation.
- **nltk**: Natural Language Processing (NLP) tasks, including tokenization, stopword removal, and stemming.
- **matplotlib** and **seaborn**: For data visualization.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Navigate into the project directory:
   ```bash
   cd sentiment-analysis-project
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open `Sentimental_analysis.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook Sentimental_analysis.ipynb
   ```

5. Run the cells sequentially to see the model training process and evaluation.

## Model Performance

The model achieves an accuracy of **90%** based on the test data. You can experiment with different machine learning models and feature extraction techniques to further improve the performance.

## Future Enhancements

- Experiment with deep learning models like **LSTM** or **BERT** for improved sentiment classification.
- Use a larger and more diverse dataset to train the model.
- Integrate the sentiment analysis model into a web application or API for real-time predictions.
