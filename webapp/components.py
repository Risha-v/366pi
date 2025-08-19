import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd

def create_metric_card(title: str, value: str, delta: str = None):
    """Create a metric display card"""
    st.metric(label=title, value=value, delta=delta)

def create_confidence_chart(result: Dict) -> go.Figure:
    """Create confidence chart for predictions"""
    if not result.get('success') or not result.get('all_probabilities'):
        fig = go.Figure()
        fig.add_annotation(text="No prediction data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    probs = result['all_probabilities']
    classes = list(probs.keys())
    confidences = [prob * 100 for prob in probs.values()]
    
    # Sort by confidence
    sorted_data = sorted(zip(classes, confidences), key=lambda x: x[1], reverse=True)
    classes, confidences = zip(*sorted_data)
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=confidences,
            orientation='h',
            marker=dict(color='steelblue'),
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

def create_class_distribution_chart(class_counts: Dict) -> go.Figure:
    """Create class distribution chart"""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig = go.Figure(data=[
        go.Bar(x=classes, y=counts, marker_color='lightblue')
    ])
    
    fig.update_layout(
        title="Dataset Class Distribution",
        xaxis_title="Classes",
        yaxis_title="Count",
        xaxis={'tickangle': 45}
    )
    
    return fig

def display_prediction_results(result: Dict, show_chart: bool = True):
    """Display prediction results with formatting"""
    if result['success']:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.success(f"**Predicted Class:** {result['predicted_class']}")
            st.info(f"**Confidence:** {result['confidence']*100:.2f}%")
            
            # Top 3 predictions
            if 'top_classes' in result:
                st.subheader("Top Predictions:")
                for i, (cls, conf) in enumerate(result['top_classes'][:3]):
                    st.write(f"{i+1}. {cls}: {conf*100:.2f}%")
        
        with col2:
            if show_chart:
                fig = create_confidence_chart(result)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

def create_batch_results_table(results: List[Dict]) -> pd.DataFrame:
    """Create results table for batch processing"""
    table_data = []
    for result in results:
        table_data.append({
            'Filename': result.get('filename', result.get('image_path', 'Unknown')),
            'Predicted Class': result.get('predicted_class', 'Error') if result['success'] else 'Failed',
            'Confidence': f"{result.get('confidence', 0)*100:.2f}%" if result['success'] else 'N/A',
            'Status': '✅ Success' if result['success'] else '❌ Failed'
        })
    
    return pd.DataFrame(table_data)

def create_progress_bar(current: int, total: int, text: str = "Processing"):
    """Create progress bar with text"""
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.write(f"{text}: {current}/{total} ({progress*100:.1f}%)")

def display_model_info(model_info: Dict):
    """Display model information in formatted cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Architecture", model_info.get('architecture', 'Unknown'))
    with col2:
        st.metric("Classes", model_info.get('num_classes', 0))
    with col3:
        st.metric("Device", model_info.get('device', 'Unknown'))
    with col4:
        st.metric("Input Size", f"{model_info.get('input_size', [224, 224])[0]}x{model_info.get('input_size', [224, 224])[1]}")

def create_download_button(data: str, filename: str, mime_type: str, label: str):
    """Create download button for data"""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )

def display_error_message(error: str):
    """Display formatted error message"""
    st.error(f"❌ Error: {error}")

def display_success_message(message: str):
    """Display formatted success message"""
    st.success(f"✅ {message}")

def display_warning_message(message: str):
    """Display formatted warning message"""
    st.warning(f"⚠️ {message}")

def display_info_message(message: str):
    """Display formatted info message"""
    st.info(f"ℹ️ {message}")
