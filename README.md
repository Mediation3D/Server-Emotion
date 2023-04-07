# Facial Emotion Recognition (Websocket Server)

![](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)

## facial emotion recognition as websockets server
- Estimate face mesh using MediaPipe(Python version).This is a sample program that recognizes facial emotion with a simple multilayer perceptron using the detected key points that returned from mediapipe.Although this model is 97% accurate, there is no generalization due to too little training data.
- the project is implement from https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe to use in facial emotion recognition

## Requirements
- mediapipe 0.8.9 : `pip install mediapipe`
- OpenCV 4.5.4 or Later : `pip install opencv-python`
- Tensorflow 2.7.0 or Later : `pip install tensorflow`
- Websockets : `pip install websockets`

## Usage
1. Clone the project
2. Install requirements
3. Start server `python server.py`

## Archives Script from ([Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe))
### Collect_from_image.py
This script will collect the keypoints from image dataset(.jpg). you can change your dataset directory to collect data.It will use your folder name to label.

### Collect_from_webcam.py
This script will collect the keypoints from your camera. press 'k' to enter the mode to save key points that show 'Record keypoints mode' then press '0-9' as label. the key points will be added to "model/keypoint_classifier/keypoint.csv". 

### main.py
This is a sample program for inference.it will use keypoint_classifier.tflite as model to predict your emotion.

### libs/emotion_recognition/training.ipynb
This is a model training script for facial emotion recognition.

## Credits

- **Kazuhito Takahashi**
  - Twitter : [高橋 かずひと@闇のパワポLT職人 (@KzhtTkhs)](https://twitter.com/KzhtTkhs)
  - Github : [Kazuhito00](https://github.com/Kazuhito00)
  - Project : [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

- **Rattasart Sakunrat**
  - Github : [REWTAO](https://github.com/REWTAO)
  - Project : [REWTAO/Facial-emotion-recognition-using-mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe)
