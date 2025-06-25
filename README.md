
# Lung Disease Detection from Chest X-ray

This is a **Streamlit web app** that predicts **Pneumonia** or **Normal** from chest X-ray images using a trained **Convolutional Neural Network (CNN)** model built with TensorFlow.Users can upload a chest X-ray image and get an instant prediction with model confidence.



## Features

- Upload chest X-ray images (`.jpg`, `.jpeg`, `.png`)
- Predicts Pneumonia or Normal
- Shows prediction confidence
- Built with Streamlit and TensorFlow
- Easy to run locally


## Project Structure
lung_detection_streamlit:
- app.py # Streamlit app
- lung_disease_model.h5 # Trained ML model
- sample_image.jpg # test image
- README.md # Project documentation
## Installation

Install my-project with npm:

```bash
  git clone https://github.com/PoojithaU-4/lung-disease-detection.git
cd lung-disease-detection
```
  Install Required Libraries:
  ```bash
  pip install streamlit tensorflow pillow numpy
```
streamlit → for the web app

tensorflow → for the Machine Learning model

pillow → for image input and preprocessing

numpy → for Machine Learning data handling 

Run the app:
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.
## Model Info
- Input Size: 150×150 pixels RGB

- Model: Convolutional Neural Network (3 conv layers + dense)

- Dataset: Chest X-ray Pneumonia Dataset (Kaggle)
## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pillow
- NumPy


## Author

 Poojitha Udutha
 - [Github](https://github.com/PoojithaU-4)
 - [Linkedin](https://www.linkedin.com/in/poojitha9023/)



## Credits
- TensorFlow Documentation
- Streamlit Docs
- Kaggle Datasets