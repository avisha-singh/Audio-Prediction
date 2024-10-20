# Genre Classification Using Audio

## Overview
This project demonstrates **Genre Classification** using machine learning techniques applied to audio files. We classify songs into 10 distinct genres based on audio features. The dataset consists of 100 songs per genre, each divided into segments for detailed analysis. The project also includes a web interface where users can upload audio files for classification.

## Dataset
- **Source**: The dataset contains 100 songs for each of 10 genres.
- **Format**: Each song is 30 seconds long, divided into 10 parts (3 seconds each).
- **Features**: Along with the audio files, feature-extracted CSV files are provided, which are used for training and testing the classification models.

## Classification Techniques
We applied various machine learning classifiers to compare their performance:

1. **K-Nearest Neighbors (KNN)**
   - K = 13 and 5 were tested; best accuracy was found at **K = 1** with an accuracy score of **0.821**.
   
2. **Bayes Classifier**
   - Applied to Unimodal and Multimodal Gaussian Density Models. Accuracy scores were **0.531** and **0.158**, respectively.
   
3. **Decision Tree**
   - A simple tree structure with an accuracy score of **0.600**.
   
4. **Random Forest**
   - Ensemble learning method with a **0.843** accuracy score.
   
5. **XGBoost**
   - Boosting model with the highest accuracy score of **0.882**.

### Methodology
1. **Data Preprocessing**
   - Outliers were removed, and the dataset was split into training (70%) and testing (30%) sets. Features were extracted and normalized.

2. **Feature Extraction Using Librosa**
   - Audio features like chroma, spectral contrast, etc., were extracted using the Librosa library. These features were normalized and fed into our classifier models.

3. **Web Application (Flask)**
   - The project includes a simple web interface where users can upload `.wav` audio files. The most accurate classifier (XGBoost) predicts the genre of the uploaded audio file. The web pages are styled using HTML and CSS.

## Web Interface
- The web app consists of two HTML pages:
  1. An upload form for selecting `.wav` audio files.
  2. A results page displaying the predicted genre.
- Flask is used to connect the machine learning models to the web interface.

## Results
The **XGBoost** classifier yielded the highest accuracy of **88.2%**, making it the preferred model for our genre classification system.

## Future Scope
- Incorporating **lyrics-based classification** using speech recognition.
- Decomposing the songs into their instrumental components for a more detailed analysis.

## How to Run the Project
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
