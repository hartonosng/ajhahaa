import streamlit as st
import pandas as pd
from streamlit.components.v1 import html

# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)

# Function to render the DataFrame as an HTML table with copy functionality
def render_df_with_copy(df):
    html_table = df.to_html(classes='table table-striped', index=False)
    
    copy_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const cells = document.querySelectorAll('td');
        cells.forEach(cell => {
            cell.addEventListener('click', function() {
                const text = this.innerText;
                navigator.clipboard.writeText(text).then(() => {
                    console.log('Copied to clipboard:', text);
                }).catch(err => {
                    console.error('Failed to copy:', err);
                });
            });
        });
    });
    </script>
    """
    
    # Render HTML and JavaScript
    html(f"""
    <div>
        {html_table}
    </div>
    {copy_script}
    """)

st.write("### Click on a cell to copy its value:")
render_df_with_copy(df)
