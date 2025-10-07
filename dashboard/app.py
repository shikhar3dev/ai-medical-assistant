"""Streamlit Dashboard for Federated Learning Disease Prediction"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import yaml
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from federated.models import DiseasePredictor
from preprocessing.preprocessor import DataPreprocessor
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from evaluation.metrics import calculate_metrics


# Page configuration
st.set_page_config(
    page_title="FL Disease Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration file."""
    config_path = Path("configs/fl_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model_and_data(config):
    """Load trained model and test data."""
    try:
        # Load preprocessor or create feature names
        preprocessor_path = Path(config['paths']['processed_data_dir']) / "preprocessor.pkl"
        try:
            import pickle
            with open(preprocessor_path, 'rb') as f:
                preprocessor_info = pickle.load(f)
            feature_names = preprocessor_info.get('feature_names', [f'feature_{i}' for i in range(13)])
        except:
            # Default heart disease feature names
            feature_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
        
        # Load test data
        test_path = Path(config['paths']['processed_data_dir']) / "test_data.pt"
        test_data = torch.load(test_path, weights_only=False)
        X_test = test_data['X']
        y_test = test_data['y']
        
        # Load model
        model_path = Path(config['paths']['models_dir']) / "best_model.pt"
        checkpoint = torch.load(model_path, weights_only=False)
        
        # Create model
        input_size = X_test.shape[1]
        model = DiseasePredictor(
            input_size=input_size,
            hidden_layers=config['model']['hidden_layers'],
            output_size=config['model']['output_size'],
            dropout=config['model']['dropout'],
            batch_norm=config['model']['batch_norm'],
            activation=config['model']['activation']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, X_test, y_test, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Privacy-Preserving Federated Learning Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Early Disease Prediction with Explainable AI")
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Performance", "Explainability", "Privacy Analysis", "Patient Prediction"]
    )
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data(config)
    
    if model is None:
        st.warning("⚠️ Model not found. Please train the model first.")
        st.info("Run the following commands to train the model:")
        st.code("""
# 1. Download and partition data
python preprocessing/data_loader.py --download
python preprocessing/partitioner.py --num-clients 3

# 2. Start FL server (in one terminal)
python federated/server.py --rounds 50

# 3. Start FL clients (in separate terminals)
python federated/client.py --client-id 0
python federated/client.py --client-id 1
python federated/client.py --client-id 2
        """)
        return
    
    # Route to pages
    if page == "Overview":
        show_overview(config, model, X_test, y_test, feature_names)
    elif page == "Model Performance":
        show_model_performance(model, X_test, y_test)
    elif page == "Explainability":
        show_explainability(model, X_test, y_test, feature_names, config)
    elif page == "Privacy Analysis":
        show_privacy_analysis(config)
    elif page == "Patient Prediction":
        show_patient_prediction(model, feature_names, config)


def show_overview(config, model, X_test, y_test, feature_names):
    """Show overview page."""
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("FL Rounds", config['federated']['num_rounds'])
    with col2:
        st.metric("Clients", config['federated']['min_clients'])
    with col3:
        st.metric("Features", len(feature_names))
    with col4:
        st.metric("Test Samples", len(X_test))
    
    st.markdown("---")
    
    # Model architecture
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Architecture")
        arch_data = {
            "Layer": ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"],
            "Size": [
                len(feature_names),
                config['model']['hidden_layers'][0],
                config['model']['hidden_layers'][1],
                config['model']['hidden_layers'][2],
                config['model']['output_size']
            ]
        }
        st.table(pd.DataFrame(arch_data))
    
    with col2:
        st.subheader("🔒 Privacy Configuration")
        privacy_data = {
            "Parameter": ["Epsilon (ε)", "Delta (δ)", "Max Grad Norm", "DP Enabled"],
            "Value": [
                config['privacy']['epsilon'],
                f"{config['privacy']['delta']:.0e}",
                config['privacy']['max_grad_norm'],
                "✅" if config['privacy']['enable_dp'] else "❌"
            ]
        }
        st.table(pd.DataFrame(privacy_data))
    
    st.markdown("---")
    
    # Quick performance summary
    st.subheader("⚡ Quick Performance Summary")
    
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        y_true = y_test.numpy().flatten()
        
        metrics = calculate_metrics(y_true, preds, probs)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("AUROC", f"{metrics['auroc']:.3f}")
    with col3:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    with col4:
        st.metric("Precision", f"{metrics['precision']:.3f}")


def show_model_performance(model, X_test, y_test):
    """Show model performance page."""
    st.header("📈 Model Performance Analysis")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        y_true = y_test.numpy().flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, preds, probs)
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")
        st.metric("Recall", f"{metrics['recall']:.4f}")
    
    with col2:
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        st.metric("Specificity", f"{metrics['specificity']:.4f}")
        st.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
    
    with col3:
        st.metric("AUROC", f"{metrics['auroc']:.4f}")
        st.metric("AUPRC", f"{metrics['auprc']:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, preds)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['No Disease', 'Disease'],
            y=['No Disease', 'Disease'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probs[y_true == 0],
            name='True Negative',
            opacity=0.7,
            marker_color='blue'
        ))
        fig.add_trace(go.Histogram(
            x=probs[y_true == 1],
            name='True Positive',
            opacity=0.7,
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Prediction Probability Distribution",
            xaxis_title="Predicted Probability",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, probs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {metrics["auroc"]:.3f})',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def show_explainability(model, X_test, y_test, feature_names, config):
    """Show explainability page."""
    st.header("🔍 Model Explainability")
    
    # Load or generate explanations
    explanations_dir = Path(config['paths']['explanations_dir'])
    
    # Feature importance
    st.subheader("Feature Importance")
    
    try:
        # Try to load pre-computed importance
        importance_path = explanations_dir / "feature_importance.csv"
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            
            # Plot
            fig = px.bar(
                importance_df.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 15 Most Important Features (SHAP)',
                labels={'importance': 'SHAP Importance', 'feature': 'Feature'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not computed yet. Run SHAP explainer first.")
            if st.button("Generate SHAP Explanations"):
                with st.spinner("Generating SHAP explanations..."):
                    # Generate explanations
                    background_data = X_test[:100].numpy()
                    explainer = SHAPExplainer(model, background_data, feature_names)
                    importance_df = explainer.get_feature_importance(X_test[:200].numpy())
                    
                    # Save
                    importance_path.parent.mkdir(parents=True, exist_ok=True)
                    importance_df.to_csv(importance_path, index=False)
                    
                    st.success("Explanations generated!")
                    st.experimental_rerun()
    
    except Exception as e:
        st.error(f"Error loading explanations: {e}")
    
    st.markdown("---")
    
    # Individual instance explanation
    st.subheader("Individual Instance Explanation")
    
    instance_idx = st.slider("Select Instance", 0, len(X_test) - 1, 0)
    
    if st.button("Explain Instance"):
        with st.spinner("Generating explanation..."):
            # Get instance
            x = X_test[instance_idx:instance_idx+1].numpy()
            
            # Get prediction
            with torch.no_grad():
                output = model(torch.FloatTensor(x))
                prob = torch.sigmoid(output).item()
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Probability", f"{prob:.4f}")
            with col2:
                pred_class = "Disease" if prob >= 0.5 else "No Disease"
                st.metric("Predicted Class", pred_class)
            
            # Feature values
            st.subheader("Feature Values")
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': x.flatten()
            })
            st.dataframe(feature_df, height=300)


def show_privacy_analysis(config):
    """Show privacy analysis page."""
    st.header("🔒 Privacy Analysis")
    
    # Privacy budget
    st.subheader("Privacy Budget")
    
    epsilon = config['privacy']['epsilon']
    delta = config['privacy']['delta']
    num_rounds = config['federated']['num_rounds']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total ε (Epsilon)", f"{epsilon:.2f}")
    with col2:
        st.metric("δ (Delta)", f"{delta:.0e}")
    with col3:
        st.metric("ε per Round", f"{epsilon/num_rounds:.4f}")
    
    # Privacy budget consumption
    st.subheader("Privacy Budget Consumption")
    
    rounds = list(range(1, num_rounds + 1))
    epsilon_spent = [epsilon * r / num_rounds for r in rounds]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds,
        y=epsilon_spent,
        mode='lines+markers',
        name='ε Spent',
        line=dict(color='red', width=2)
    ))
    fig.add_hline(y=epsilon, line_dash="dash", line_color="green", 
                  annotation_text="Total Budget")
    
    fig.update_layout(
        title="Privacy Budget Consumption Over Rounds",
        xaxis_title="Federated Learning Round",
        yaxis_title="Cumulative ε",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Privacy parameters
    st.subheader("Privacy Parameters")
    
    privacy_params = {
        "Parameter": [
            "Differential Privacy",
            "Max Gradient Norm",
            "Noise Multiplier",
            "Secure Aggregation",
            "Accounting Method"
        ],
        "Value": [
            "✅ Enabled" if config['privacy']['enable_dp'] else "❌ Disabled",
            config['privacy']['max_grad_norm'],
            config['privacy'].get('noise_multiplier', 'Auto'),
            "✅ Enabled" if config['privacy']['enable_secure_aggregation'] else "❌ Disabled",
            config['privacy']['accounting_method'].upper()
        ]
    }
    st.table(pd.DataFrame(privacy_params))


def show_patient_prediction(model, feature_names, config):
    """Show patient prediction page."""
    st.header("👤 Patient Risk Prediction")
    
    st.write("Enter patient information to predict disease risk:")
    
    # Create input form
    with st.form("patient_form"):
        st.subheader("Patient Information")
        
        # Create inputs for each feature (simplified)
        inputs = {}
        
        # Use columns for better layout
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            if i % 2 == 0:
                with col1:
                    inputs[feature] = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=0.0,
                        step=0.1
                    )
            else:
                with col2:
                    inputs[feature] = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=0.0,
                        step=0.1
                    )
        
        submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        # Create input tensor
        x = torch.FloatTensor([[inputs[f] for f in feature_names]])
        
        # Get prediction
        with torch.no_grad():
            output = model(x)
            prob = torch.sigmoid(output).item()
        
        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Disease Risk", f"{prob*100:.1f}%")
        
        with col2:
            risk_level = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
            st.metric("Risk Level", risk_level)
        
        with col3:
            pred_class = "Disease" if prob >= 0.5 else "No Disease"
            st.metric("Prediction", pred_class)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Disease Risk (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
