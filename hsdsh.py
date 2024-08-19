import streamlit as st
import shap
import plotly.graph_objects as go
import numpy as np

# Example SHAP values (using a simple model)
# Assuming `model` is your trained model and `X` is your dataset

# Generate a random dataset for example purposes
X = np.random.rand(1, 10)  # 1 instance with 10 features

# Generate SHAP values using a random forest (for example)
explainer = shap.Explainer(lambda x: x.sum(axis=1, keepdims=True), X)
shap_values = explainer(X)

# Function to plot SHAP waterfall using Plotly
def plot_shap_waterfall(shap_values, feature_names=None):
    base_value = shap_values.base_values[0]
    shap_values = shap_values.values[0]

    # If feature names are not provided, generate generic names
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(len(shap_values))]

    # Sort SHAP values and feature names by absolute value of SHAP values
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    shap_values = shap_values[sorted_indices]
    feature_names = np.array(feature_names)[sorted_indices]

    # Calculate cumulative sum to determine the flow
    cumulative = np.cumsum(shap_values)

    # Initialize figure
    fig = go.Figure()

    # Add bars for each SHAP value
    for i in range(len(shap_values)):
        fig.add_trace(go.Bar(
            x=[feature_names[i]],
            y=[shap_values[i]],
            orientation='v',
            marker=dict(color='red' if shap_values[i] < 0 else 'green'),
            name=f"{feature_names[i]}",
            showlegend=False
        ))

    # Add line for cumulative effect
    fig.add_trace(go.Scatter(
        x=feature_names,
        y=cumulative,
        mode='lines+markers',
        marker=dict(color='blue'),
        name='Cumulative effect'
    ))

    # Add a horizontal line at the base value
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=base_value,
        x1=len(feature_names)-0.5,
        y1=base_value,
        line=dict(color="black", width=2, dash="dash")
    )

    # Update layout
    fig.update_layout(
        title="SHAP Waterfall Plot",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        template="plotly_white",
        showlegend=False
    )

    return fig

# Streamlit app
st.title("SHAP Waterfall Plot in Streamlit")

# Plot SHAP waterfall plot with generic feature names
shap_plot = plot_shap_waterfall(shap_values)

# Display the plot in Streamlit
st.plotly_chart(shap_plot)
