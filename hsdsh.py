import streamlit as st
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load example data
X, y = shap.datasets.boston()

# Train a model
model = xgb.XGBRegressor().fit(X, y)

# Explain the model's predictions using SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Select the index of the instance you want to display
index = 9  # For instance, you want to show SHAP values for index 9

# Extract SHAP values for the selected instance
shap_value = shap_values[index]
base_value = shap_value.base_values
feature_names = X.columns
shap_values_instance = shap_value.values

# Sort the SHAP values by absolute magnitude
sorted_indices = np.argsort(np.abs(shap_values_instance))[::-1]
sorted_shap_values = shap_values_instance[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Calculate cumulative sum for the waterfall effect
cumulative = np.cumsum(sorted_shap_values)

# Create Plotly figure
fig = go.Figure()

# Add bars for each SHAP value
for i in range(len(sorted_shap_values)):
    fig.add_trace(go.Bar(
        x=[sorted_feature_names[i]],
        y=[sorted_shap_values[i]],
        orientation='v',
        marker=dict(color='red' if sorted_shap_values[i] < 0 else 'green'),
        name=f"{sorted_feature_names[i]}",
        showlegend=False
    ))

# Add line for cumulative effect
fig.add_trace(go.Scatter(
    x=sorted_feature_names,
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
    x1=len(sorted_feature_names)-0.5,
    y1=base_value,
    line=dict(color="black", width=2, dash="dash")
)

# Update layout
fig.update_layout(
    title=f"SHAP Waterfall Plot for Instance {index}",
    xaxis_title="Features",
    yaxis_title="SHAP Value",
    template="plotly_white",
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=500
)

# Display the plot in Streamlit
st.title(f"SHAP Waterfall Plot for Instance {index}")
st.plotly_chart(fig)
