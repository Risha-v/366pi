# Eurusat Classifier Project
EuroSAT Land Coverifier
Overview
The EuroSAT Landifier is a powerful tool for classifying land cover types from satellite images, utilizing the state-of-the-art MobileNetV3 architecture pretrained on ImageNet.

Features
Classification of 10 land cover classes

Supports single and batch image classification

High accuracy with modern augmentation strategies

Support for model export in ONNX and TorchScript

Interactive Streamlit web application

Installation
Install dependencies:


pip install -r requirements.txt
Data Preparation
Download and preprocess the EuroSAT dataset.
python prepare_data.py

Training
Train the model:


python train_model.py --epochs 50 --batch-size 16
Exporting Model
Export trained model to ONNX and TorchScript formats:


python export_model.py --formats all
Run Web App
Start the streamlit app:


streamlit run webapp/app.py
Citation
If you use this work for research, please cite:


@inproceedings{helber2019eurosat,
  title={EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Patrick Helber and Benjamin Bischke and Andreas Dengel and Damian Borth},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}