# Emotion Detection Through Voice Using Deep Learning

This project implements **emotion detection through speech** using deep learning techniques. Leveraging datasets like RAVDESS, CREMA-D, TESS, and SAVEE, it classifies human emotions (e.g., happy, sad, angry) from audio recordings.

## Features
- Emotion classification into categories: Angry, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- Utilizes **Convolutional Neural Networks (CNNs)** for feature extraction and classification.
- Data augmentation techniques for improving model robustness.
- Balanced dataset handling to address class imbalance.

## Datasets
The following datasets are used for training and evaluation:
- [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
- [CREMA-D](https://www.kaggle.com/ejlok1/cremad)
- [TESS](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
- [SAVEE](https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee)

## Model Architecture
- **4 Convolutional Layers** with MaxPooling for feature extraction.
- Dense layers for classification.
- Dropout layers to prevent overfitting.
- Final Softmax layer with 8 outputs (one for each emotion).

## Results
- **Accuracy:** 64.72%
- Precision, recall, and F1-score provided for each class in the confusion matrix.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotion-detection-voice-dl.git
   cd emotion-detection-voice-dl
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `librosa`
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `seaborn`
   - `keras`
   - `sklearn`

3. Download the datasets and place them in the respective folders as described in the notebook.

## Usage
1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open the `Emotion_Detection.ipynb` file and follow the instructions.
   
2. To train the model:
   - Execute the "Model Training" section in the notebook.
   
3. To evaluate the model:
   - Run the "Evaluation" section, which includes accuracy, loss plots, and the confusion matrix.

## Future Improvements
- Incorporate more advanced models like transformers.
- Explore multi-modal approaches combining speech and text analysis.
- Extend the model to real-time emotion detection applications.

## Author
**Dineth Hettiarachchi**

## Contributing
Feel free to contribute! Fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License.
