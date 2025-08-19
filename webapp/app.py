import streamlit as st
import sys
import os
import gc
import logging
import time
import json
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Configure Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from configs.default import Config
    from src.inference import EuroSATClassifier
except ImportError as e:
    st.error(f"Critical import error: {str(e)}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="EuroSAT Land Cover Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class information
CLASS_INFO = {
    'AnnualCrop': {'description': 'Agricultural areas with annual crops', 'color': '#32CD32', 'icon': '🌾'},
    'Forest': {'description': 'Dense forest areas', 'color': '#228B22', 'icon': '🌲'},
    'HerbaceousVegetation': {'description': 'Natural grasslands and meadows', 'color': '#9ACD32', 'icon': '🌿'},
    'Highway': {'description': 'Major roads and transportation infrastructure', 'color': '#696969', 'icon': '🛣️'},
    'Industrial': {'description': 'Industrial complexes and factories', 'color': '#8B4513', 'icon': '🏭'},
    'Pasture': {'description': 'Grazing lands for livestock', 'color': '#90EE90', 'icon': '🐄'},
    'PermanentCrop': {'description': 'Orchards and permanent crops', 'color': '#FF6347', 'icon': '🍎'},
    'Residential': {'description': 'Urban residential areas', 'color': '#FF69B4', 'icon': '🏘️'},
    'River': {'description': 'Rivers and flowing water bodies', 'color': '#4169E1', 'icon': '🏞️'},
    'SeaLake': {'description': 'Large water bodies', 'color': '#1E90FF', 'icon': '🌊'}
}

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load and cache the model with better error handling"""
    try:
        model_path = Config.get_trained_model_path()
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        if file_size < 1:  # Less than 1MB is likely corrupted
            st.error(f"Model file appears corrupted (size: {file_size:.2f}MB)")
            return None
        
        with st.spinner("Loading model..."):
            model = EuroSATClassifier(model_path)
            st.success(f"Model loaded successfully!")
        
        return model
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"Failed to load model: {error_msg}")
        
        # Provide specific guidance based on error type
        if "Missing key(s) in state_dict" in error_msg:
            st.error("Model architecture mismatch. The model file may be from a different training run.")
        elif "No such file or directory" in error_msg:
            st.error("Model file not found. Please train the model first.")
        elif "CUDA" in error_msg:
            st.error("CUDA error. Try running on CPU or check GPU availability.")
        
        return None

def create_confidence_chart(result: dict) -> go.Figure:
    """Create confidence chart"""
    if not result.get('success'):
        fig = go.Figure()
        fig.add_annotation(text="No prediction data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    top_classes = result.get('top_classes', [])[:5]
    classes = [cls for cls, _ in top_classes]
    confidences = [prob * 100 for _, prob in top_classes]
    colors = [CLASS_INFO.get(cls, {}).get('color', '#3498db') for cls in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=confidences,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Confidence (%)",
        yaxis_title="Classes",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def show_dashboard(model):
    """Dashboard page"""
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Status", "Loaded" if model else "Not Available", 
                 delta="Ready" if model else "Error")
    
    with col2:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device, delta=f"{torch.cuda.device_count()} GPUs" if torch.cuda.is_available() else "")
    
    with col3:
        st.metric("Classes", len(Config.CLASSES))
    
    with col4:
        st.metric("Image Size", f"{Config.IMAGE_SIZE[0]}x{Config.IMAGE_SIZE[1]}")
    
    # Quick test section
    st.subheader("Quick Test")
    uploaded_file = st.file_uploader("Upload an image for quick classification", 
                                    type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file and model:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            with st.spinner("Classifying..."):
                temp_path = f"temp_{int(time.time())}.jpg"
                image.save(temp_path)
                
                try:
                    result = model.predict_image(temp_path)
                    
                    if result['success']:
                        st.success(f"**Predicted Class:** {result['predicted_class']}")
                        st.info(f"**Confidence:** {result['confidence']*100:.2f}%")
                        
                        fig = create_confidence_chart(result)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

def show_single_image(model):
    """Single image classification page"""
    st.header("Single Image Classification")
    
    if not model:
        st.error("Model not available. Please check if the model is trained and saved.")
        return
    
    uploaded_file = st.file_uploader("Choose an image file", 
                                    type=['png', 'jpg', 'jpeg', 'tiff'],
                                    help="Upload a satellite image for land cover classification")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            with st.spinner("Processing image..."):
                temp_path = f"temp_{int(time.time())}_{uploaded_file.name}"
                image.save(temp_path)
                
                try:
                    result = model.predict_image(temp_path)
                    
                    if result['success']:
                        st.success(f"**Predicted Class:** {result['predicted_class']}")
                        st.info(f"**Confidence:** {result['confidence']*100:.2f}%")
                        
                        class_info = CLASS_INFO.get(result['predicted_class'], {})
                        if class_info:
                            st.markdown(f"**Description:** {class_info['description']}")
                        
                        st.subheader("Top Predictions")
                        for i, (cls, conf) in enumerate(result['top_classes'][:5]):
                            icon = CLASS_INFO.get(cls, {}).get('icon', '📍')
                            st.write(f"{i+1}. {icon} **{cls}**: {conf*100:.2f}%")
                        
                        fig = create_confidence_chart(result)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        report = {
                            'predicted_class': result['predicted_class'],
                            'confidence': result['confidence'],
                            'all_probabilities': result['all_probabilities']
                        }
                        
                        json_str = json.dumps(report, indent=2)
                        st.download_button(
                            label="Download JSON Report",
                            data=json_str,
                            file_name=f"prediction_{int(time.time())}.json",
                            mime="application/json"
                        )
                    
                    else:
                        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

def show_batch_processing(model):
    """Batch processing page"""
    st.header("Batch Image Processing")
    
    if not model:
        st.error("Model not available.")
        return
    
    uploaded_files = st.file_uploader("Choose image files", 
                                    type=['png', 'jpg', 'jpeg'],
                                    accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Batch", type="primary"):
        progress_bar = st.progress(0)
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                image = Image.open(uploaded_file).convert('RGB')
                temp_path = f"temp_batch_{i}_{uploaded_file.name}"
                image.save(temp_path)
                
                result = model.predict_image(temp_path)
                result['filename'] = uploaded_file.name
                results.append(result)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                results.append({
                    'filename': uploaded_file.name,
                    'success': False,
                    'error': str(e)
                })
        
        # Display results
        st.subheader("Batch Processing Results")
        
        # Summary metrics
        total_files = len(results)
        successful = sum(1 for r in results if r['success'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Success Rate", f"{(successful/total_files*100):.1f}%")
        
        # Results table - Fixed DataFrame construction
        df_results = []
        for result in results:
            df_results.append({
                'Filename': str(result['filename']),
                'Predicted Class': str(result.get('predicted_class', 'Error')),
                'Confidence': f"{result.get('confidence', 0)*100:.2f}%" if result.get('confidence') is not None else 'N/A',
                'Status': 'Success' if result.get('success', False) else 'Failed'
            })
        
        df = pd.DataFrame(df_results)
        # Ensure all columns are strings to avoid Arrow conversion issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        st.dataframe(df, use_container_width=True)
        
        # Download results
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name=f"batch_results_{int(time.time())}.csv",
            mime="text/csv"
        )

def show_training_dashboard():
    """Training dashboard"""
    st.header("Training Dashboard")
    
    metadata_path = Config.TRAINED_MODEL_DIR / 'training_metadata.json'
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                training_data = json.load(f)
            
            st.subheader("Training Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best F1 Score", f"{training_data.get('best_f1_score', 0):.4f}")
            with col2:
                st.metric("Training Date", training_data.get('training_date', 'Unknown')[:10])
            with col3:
                st.metric("Epochs", training_data.get('config', {}).get('epochs', 'Unknown'))
            
        except Exception as e:
            st.error(f"Failed to load training metadata: {str(e)}")
    else:
        st.warning("No training metadata found. Train a model first.")

def show_model_evaluation(model):
    """Model evaluation page"""
    st.header("Model Evaluation")
    
    if not model:
        st.error("Model not available.")
        return
    
    # Model info
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Architecture", "MobileNetV3-Large")
    with col2:
        st.metric("Parameters", "~5.4M")
    with col3:
        device = "GPU" if next(model.model.parameters()).is_cuda else "CPU"
        st.metric("Device", device)

def show_dataset_explorer():
    """Dataset explorer page"""
    st.header("Dataset Explorer")
    st.info("Dataset statistics and exploration features")

def show_settings():
    """Settings page"""
    st.header("Settings")
    
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            st.info(f"**GPU Name:** {torch.cuda.get_device_name()}")
            st.info(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        st.info(f"**PyTorch Version:** {torch.__version__}")
        st.info(f"**Python Version:** {sys.version.split()[0]}")
    
    with col2:
        st.info(f"**Batch Size:** {Config.BATCH_SIZE}")
        st.info(f"**Image Size:** {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE[1]}")
        st.info(f"**Workers:** {Config.NUM_WORKERS}")
        st.info(f"**Mixed Precision:** {'Enabled' if Config.USE_AMP else 'Disabled'}")

def show_about():
    """About page"""
    st.header("About")
    
    st.markdown("""
    ## EuroSAT Land Cover Classifier
    
    This application provides state-of-the-art satellite image classification using deep learning.
    
    ### Features
    - Advanced Architecture: MobileNetV3-Large with custom classifier
    - Robust Training: Mixed precision, data augmentation, class balancing
    - Professional UI: Multi-page interface with comprehensive features
    - Batch Processing: Handle multiple images efficiently
    """)
    
    for class_name, info in CLASS_INFO.items():
        st.write(f"**{info['icon']} {class_name}**: {info['description']}")

def main():
    # Header
    st.title("🌍 EuroSAT Land Cover Classifier")
    st.markdown("### Advanced satellite image classification using deep learning")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = [
        "Dashboard",
        "Single Image", 
        "Batch Processing",
        "Training Dashboard",
        "Model Evaluation",
        "Dataset Explorer",
        "Settings",
        "About"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # Load model
    if not st.session_state.model_loaded:
        st.session_state.model = load_model()
        st.session_state.model_loaded = True
    
    model = st.session_state.model
    
    # Page routing
    if selected_page == "Dashboard":
        show_dashboard(model)
    elif selected_page == "Single Image":
        show_single_image(model)
    elif selected_page == "Batch Processing":
        show_batch_processing(model)
    elif selected_page == "Training Dashboard":
        show_training_dashboard()
    elif selected_page == "Model Evaluation":
        show_model_evaluation(model)
    elif selected_page == "Dataset Explorer":
        show_dataset_explorer()
    elif selected_page == "Settings":
        show_settings()
    elif selected_page == "About":
        show_about()

if __name__ == "__main__":
    main()