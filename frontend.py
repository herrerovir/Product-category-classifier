# Import required libraries
import streamlit as st
import requests

# Set Streamlit page configuration
st.set_page_config(
    page_title = "Product Category Classifier",
    page_icon = "🛍️",
    layout = "centered"
)

# Custom CSS
st.markdown("""
    <style>
    /* Input text */
    textarea {
        background-color: #2f2f2f !important;
        color: #ffffff !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    /* Prediction box */
    .prediction-box {
        background-color: #2f2f2f !important;
        color: #ffffff !important;
        border: 1px solid #886ce4 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
            
    .prediction-box p {
        display: flex;
        justify-content: flex-start;
        gap: 0.5rem;
        margin: 0.3rem 0;
        font-size: 1.1rem;
        line-height: 1.4;
    }
    
    /* Classify button */
    div.stButton > button {
        background-color: #886ce4 !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 10px rgba(136, 108, 228, 0.3);
    }

    div.stButton > button:hover {
        background-color: #a48eff !important;
        transform: scale(1.03);
        cursor: pointer;
    }
    
    /* Error message */
    .custom-error {
        background-color: #cfcae1 !important;
        color: #6840ec !important;
        border-left: 10px solid #886ce4 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        margin-top: 1rem !important;
    }
            
    </style>
""", unsafe_allow_html = True)

# Title and description
st.title("🛍️ Product Category Classifier")
st.markdown("Classify product descriptions into categories using a fine-tuned DistilBERT model.")

# Text input
text_input = st.text_area(
    "Enter product description below:",
    placeholder = "e.g., Wireless Bluetooth headphones with noise cancellation...",
    height = 150
)

# API endpoint (adjust this in HF Spaces if needed)
API_URL = "http://localhost:8000/predict"

# Predict button
if st.button("Classify"):
    if not text_input.strip():
        st.markdown("""
            <div class = "custom-error">
            <strong>Please enter a product description to classify.</strong>
            </div>
        """, unsafe_allow_html = True)

    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(API_URL, json = {"text": text_input})
                response.raise_for_status()
                result = response.json()

                # Display prediction
                st.markdown(f"""
                <div class = "prediction-box">
                    <p><span>Predicted Category:</span> <strong>{result['label']}</strong></p>
                    <p><span>Confidence:</span> <strong>{result['confidence'] * 100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html = True)

            except Exception as e:
                st.markdown(f"""
                    <div class="custom-error">
                        <strong>Failed to get prediction from backend.</strong><br>
                        <details>
                            <summary style="cursor: pointer; margin-top: 0.5rem;">Show technical details</summary>
                            <pre>{str(e).replace('<', '&lt;').replace('>', '&gt;')}</pre>
                        </details>
                    </div>
                """, unsafe_allow_html = True)