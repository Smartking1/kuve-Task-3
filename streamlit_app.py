import streamlit as st
import sys
from pathlib import Path
import yaml
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.preprocessing.text_processor import TextProcessor
from src.model.loader import ModelLoader
from src.inference.predictor import TextClassificationPredictor


# Page configuration
st.set_page_config(
    page_title="Text Classification Tester",
    page_icon="🎯",
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
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .positive {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .neutral {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_components():
    """Load model, tokenizer, and create predictor (cached)."""
    try:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model package
        model_loader = ModelLoader(config['model'])
        
        # Only use the model package
        model, tokenizer, threshold = model_loader.load_model_package(
            config['model']['package_path']
        )
        # Get max_tokens from config or set default
        max_tokens = config['preprocessing'].get('max_tokens', 100)
        
        # Initialize text processor
        text_processor = TextProcessor(
            config['preprocessing'],
            vectorizer=tokenizer  # pass tokenizer as vectorizer
        )
        
        # Initialize predictor
        predictor = TextClassificationPredictor(
            model=model,
            text_processor=text_processor,
            threshold=threshold,
            config=config['inference']
        )
        
        # Get model info
        model_info = {
            'model_type': type(model).__name__,
            'threshold': threshold,
            'max_tokens': max_tokens,
            'vocab_size': len(tokenizer.word_index) if hasattr(tokenizer, 'word_index') else None,
            'is_keras': hasattr(model, 'predict') and not hasattr(model, 'predict_proba')
        }
        
        return predictor, model_info, config
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


def create_probability_chart(probabilities):
    """Create a bar chart for class probabilities."""
    if not probabilities:
        return None
    
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Color based on sentiment
    colors = ['#dc3545' if label == 'negative' else '#28a745' for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.2%}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig


def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence level."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#ffe6e6"},
                {'range': [50, 75], 'color': "#fff9e6"},
                {'range': [75, 100], 'color': "#e6f7ff"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🎯 Text Classification Tester</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        predictor, model_info, config = load_model_components()
    
    if predictor is None:
        st.error("❌ Failed to load model. Please check your configuration and model files.")
        st.info("Make sure you have:\n- config.yaml in the current directory\n- Model files in models/ folder")
        st.stop()
    
    # Success message
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", model_info['model_type'])
        st.metric("Decision Threshold", f"{model_info['threshold']:.2f}")
        st.metric("Max Tokens", model_info['max_tokens'])
        
        if model_info['vocab_size']:
            st.metric("Vocabulary Size", f"{model_info['vocab_size']:,}")
        
        st.info(f"🤖 Keras Model: {model_info['is_keras']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        custom_threshold = st.slider(
            "Custom Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(model_info['threshold']),
            step=0.05,
            help="Adjust the decision threshold for classification"
        )
        
        show_probabilities = st.checkbox("Show Probabilities", value=True)
        show_charts = st.checkbox("Show Charts", value=True)
        show_processing = st.checkbox("Show Text Processing", value=False)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📚 Batch Prediction", "📊 Analysis"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Text Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            text_input = st.text_area(
                "Enter text to classify:",
                height=150,
                placeholder="Type or paste your text here...",
                help="Enter the text you want to classify"
            )
            
            # Example texts
            st.write("**Quick Examples:**")
            example_cols = st.columns(3)
            
            examples = [
                "This movie was absolutely fantastic!",
                "Terrible film, complete waste of time.",
                "The acting was great but confusing plot."
            ]
            
            for idx, (col, example) in enumerate(zip(example_cols, examples)):
                if col.button(f"Example {idx+1}", key=f"ex_{idx}", use_container_width=True):
                    text_input = example
                    st.rerun()
        
        with col2:
            st.write("**Quick Actions:**")
            predict_button = st.button("🎯 Predict", type="primary", use_container_width=True)
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_button:
                st.rerun()
        
        if predict_button and text_input:
            with st.spinner("🔮 Classifying..."):
                try:
                    # Make prediction
                    result = predictor.predict_single(text_input, return_probabilities=True)
                    
                    # Apply custom threshold if different
                    if custom_threshold != model_info['threshold']:
                        prob_positive = result.get('probability_positive', result['confidence'])
                        result['predicted_class'] = int(prob_positive >= custom_threshold)
                        result['label'] = 'positive' if result['predicted_class'] == 1 else 'negative'
                        result['confidence'] = prob_positive if result['predicted_class'] == 1 else (1 - prob_positive)
                    
                    # Show processed text if enabled
                    if show_processing:
                        with st.expander("🔍 Text Processing Details"):
                            processed = predictor.text_processor.preprocess(text_input)
                            st.text(f"Original:\n{text_input[:200]}...")
                            st.text(f"\nProcessed:\n{processed[:200]}...")
                    
                    # Determine style based on label
                    label_lower = str(result['label']).lower()
                    if 'positive' in label_lower:
                        box_class = "positive"
                        emoji = "😊"
                    elif 'negative' in label_lower:
                        box_class = "negative"
                        emoji = "😞"
                    else:
                        box_class = "neutral"
                        emoji = "😐"
                    
                    # Display result
                    st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
                    st.subheader(f"🎯 Prediction Result {emoji}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Class", f"{result['label'].title()} {emoji}")
                    col2.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    if 'probability_positive' in result:
                        col3.metric("Prob(Positive)", f"{result['probability_positive']:.2%}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show charts
                    if show_charts:
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            if show_probabilities and 'probabilities' in result:
                                fig = create_probability_chart(result['probabilities'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        with chart_col2:
                            fig = create_confidence_gauge(result['confidence'])
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed probabilities
                    if show_probabilities and 'probabilities' in result:
                        with st.expander("📊 Detailed Probabilities"):
                            prob_df = pd.DataFrame([
                                {'Class': k.title(), 'Probability': v, 'Percentage': f"{v:.2%}"}
                                for k, v in result['probabilities'].items()
                            ]).sort_values('Probability', ascending=False)
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Text Classification")
        
        # Option 1: Text area input
        st.subheader("Option 1: Enter Multiple Texts")
        batch_text = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="Text 1\nText 2\nText 3\n...",
            help="Enter each text on a new line"
        )
        
        # Option 2: File upload
        st.subheader("Option 2: Upload File")
        uploaded_file = st.file_uploader(
            "Upload a text file (one text per line) or CSV",
            type=['txt', 'csv'],
            help="Upload a .txt file with one text per line, or a .csv file with a 'text' column"
        )
        
        batch_predict_button = st.button("🎯 Predict Batch", type="primary")
        
        if batch_predict_button:
            texts = []
            
            # Get texts from input
            if batch_text:
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
            
            # Get texts from file
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts.extend(df['text'].dropna().tolist())
                    else:
                        st.error("❌ CSV file must have a 'text' column")
                else:
                    content = uploaded_file.read().decode('utf-8')
                    texts.extend([line.strip() for line in content.split('\n') if line.strip()])
            
            if texts:
                with st.spinner(f"🔮 Classifying {len(texts)} texts..."):
                    try:
                        # Make batch prediction
                        results = predictor.predict_batch(texts, return_probabilities=True)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame([
                            {
                                'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                                'Predicted': r['label'].title(),
                                'Confidence': f"{r['confidence']:.2%}",
                                'Prob(Positive)': f"{r.get('probability_positive', 0):.2%}",
                                'Confident?': '✅' if r['is_confident'] else '❌'
                            }
                            for r in results
                        ])
                        
                        # Display summary
                        st.success(f"✅ Processed {len(texts)} texts")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Count predictions by class
                        pos_count = sum(1 for r in results if r['label'] == 'positive')
                        neg_count = sum(1 for r in results if r['label'] == 'negative')
                        
                        col1.metric("Total Texts", len(texts))
                        col2.metric("Positive", pos_count)
                        col3.metric("Negative", neg_count)
                        col4.metric("Confident", sum(1 for r in results if r['is_confident']))
                        
                        # Display results table
                        st.subheader("Results")
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Class distribution chart
                        if show_charts:
                            st.subheader("Class Distribution")
                            class_counts = pd.Series([r['label'] for r in results]).value_counts()
                            fig = px.pie(
                                values=class_counts.values,
                                names=[n.title() for n in class_counts.index],
                                title="Prediction Distribution",
                                color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Batch prediction failed: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Please enter texts or upload a file.")
    
    # Tab 3: Analysis
    with tab3:
        st.header("Model Analysis & Testing")
        
        st.subheader("🔍 Threshold Analysis")
        
        analysis_text = st.text_input(
            "Enter text for threshold analysis:",
            value="This is a sample text for analysis.",
            help="Analyze how different thresholds affect the prediction"
        )
        
        if st.button("Analyze Thresholds", use_container_width=True):
            with st.spinner("🔍 Analyzing..."):
                try:
                    thresholds_list = predictor.evaluate_thresholds(analysis_text)
                    
                    if thresholds_list:
                        df = pd.DataFrame(thresholds_list)
                        
                        # Create threshold chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['threshold'],
                            y=df['probability_positive'],
                            mode='lines',
                            name='Probability (Positive)',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['threshold'],
                            y=df['threshold'],
                            mode='lines',
                            name='Threshold Line',
                            line=dict(color='red', dash='dash', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Probability vs Threshold",
                            xaxis_title="Threshold",
                            yaxis_title="Value",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("ℹ️ Threshold analysis is only available for binary classification.")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
        
        st.markdown("---")
        
        st.subheader("📈 Performance Tips")
        st.info("""
        **Tips for better predictions:**
        - ✅ Ensure input text is similar to training data
        - ✅ Longer texts generally provide more context
        - ✅ Check confidence scores - low confidence may indicate uncertain predictions
        - ✅ Adjust threshold based on your precision/recall requirements
        - ✅ The model works best with movie reviews and similar sentiment-heavy text
        """)


if __name__ == "__main__":
    main()