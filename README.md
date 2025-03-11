# Email/SMS Spam Detector

## Project Overview
This project is an Email/SMS Spam Classifier built using Natural Language Processing (NLP) techniques and deployed as a web application using Streamlit. The classifier can analyze a given message and predict whether it is spam or not.

## Goals
- Develop a machine learning model to detect spam messages with high accuracy.
- Implement text preprocessing techniques to clean and normalize input messages.
- Deploy the model as an interactive web application for real-time predictions.
- Ensure an efficient and lightweight solution for ease of use.

## Tools & Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - `streamlit` - For building the web application.
  - `nltk` - For text preprocessing (tokenization, stopwords removal, stemming).
  - `sklearn` - For model training and TF-IDF vectorization.
- **Machine Learning Model:** Trained on a labeled dataset of spam and ham (non-spam) messages.
- **Deployment:** Streamlit-based interactive UI.

## Files & Directories
- `app.py` - Main script to run the Streamlit app.
- `model.pkl` - Trained machine learning model (saved using Pickle).
- `vectorizer.pkl` - TF-IDF vectorizer for text transformation.
- `spam.csv` - Dataset used for training the model.
- `requirements.txt` - Dependencies required to run the project.
- `setup.sh` - Configuration script for deployment.
- `nltk.txt` - List of required NLTK components.
- `sms-spam-detection.ipynb` - Jupyter notebook with data preprocessing, model training, and evaluation.

## Installation & Setup
### Prerequisites
Ensure you have Python installed on your system.

### Steps to Run Locally
1. Clone this repository.
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the required NLTK resources:
   ```sh
   python -m nltk.downloader stopwords punkt
   ```
4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## How It Works
1. The user enters a message into the text area.
2. The message is preprocessed (tokenization, stopword removal, stemming).
3. The transformed text is vectorized using the TF-IDF method.
4. The trained model predicts whether the message is spam or not.
5. The result is displayed on the Streamlit interface.

## Future Improvements
- Improve accuracy with deep learning models like LSTMs or transformers.
- Expand the dataset for better generalization.
- Deploy the application using cloud platforms for wider accessibility.

## Author
Developed by **Salma Elmasry**

