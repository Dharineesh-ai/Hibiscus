🌿 Hibiscus Leaf Disease Prediction using Deep Learning

This project focuses on identifying diseases in hibiscus plant leaves using Convolutional Neural Networks (CNN). The primary goal is to detect early signs of diseases such as Leaf Spot, Powdery Mildew, Yellow Spot Virus, and Root Rot using image classification techniques and deploy the model for real-time use.

📌 Project Features
✅ Deep Learning-based Image Classification

✅ Preprocessing and Augmentation

✅ Hybrid Model: ACO (Ant Colony Optimization) + CNN

✅ Mobile Deployment with Transfer Learning (MobileNetV2/MobileNetV3)

✅ Explainable AI (Grad-CAM) for Visualization

✅ Real-time Leaf Detection Capability

🗂 Dataset
Source: Kaggle Dataset (custom hibiscus leaf dataset)

Structure:

Edit
├── dataset/
    ├── train/
    │   ├── fresh/
    │   └── diseased/
    └── test/
        ├── fresh/
        └── diseased/
Images: ~150 images of hibiscus leaves (both healthy and diseased)

⚙️ Technologies Used
Python 3.8

TensorFlow / Keras

OpenCV

Matplotlib, NumPy, Pandas


🧠 Model Architecture
Custom CNN with tuned hyperparameters

ACO-CNN hybrid model to enhance accuracy

MobileNetV2 and MobileNetV3 for transfer learning and mobile deployment

Grad-CAM for visualization of learned features

📈 Results
Accuracy: 92% (on test data using ACO-CNN)

Performance evaluated using confusion matrix, precision, recall, F1-score

📱 Mobile Deployment (Optional)
Converted the trained model using TensorFlow Lite.

Deployed using an Android application for offline leaf scanning.

🔬 Explainable AI
Integrated Grad-CAM to highlight regions influencing model predictions.

Helps in understanding model decisions and building trust in AI.

📚 Future Work
Expand dataset with more leaf samples

Include severity level prediction

Integration with smart farming apps

IoT-based real-time monitoring setup
