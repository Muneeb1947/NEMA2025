import streamlit as st
import os
import joblib

# Define model path relative to the working directory
model_path = os.path.join(os.getcwd(), "models", "xgb_tree_model2.joblib")

# Function to load the model with caching


@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(
            "Model file not found! Ensure 'xgb_tree_model2.joblib' exists in the 'models' folder.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load model
model = load_model()

# Streamlit UI
st.title("✈️ Flight Delay Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Flight Delay Prediction Dashboard**!  
    Use the sidebar to navigate through different pages and analyze flight delay predictions.
    """
)

# Sidebar navigation
st.sidebar.header("Navigation")
st.sidebar.write("Use the sidebar to explore different analysis pages.")

# Model Status Indicator
if model:
    st.sidebar.success("✅ Model Loaded Successfully")
else:
    st.sidebar.warning("⚠️ Model Not Loaded")
