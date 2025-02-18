import streamlit as st
import os
 
 
# Set up the page configuration
st.set_page_config(
    page_title="Research & Plot Agent Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
 
# Custom CSS styling for a modern look
st.markdown(
    """
    <style>
    /* Set the background color for the main container */
    .reportview-container {
        background: #f0f2f6;
    }
    /* Style for the header */
    .header {
        text-align: center;
        padding: 1rem;
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    /* Style for any container with information */
    .info-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
 
# Main header section
st.markdown('<div class="header"><h1>Research & Plot Agent Dashboard</h1></div>', unsafe_allow_html=True)
 
 
# Create two columns for layout: one for the image and one for additional information
col1, col2 = st.columns([2, 1])
 
 
with col1:
    st.markdown('<div class="info-container">', unsafe_allow_html=True)
    plot_path = "output/plot.png"
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Generated Plot", use_column_width=True)
    else:
        st.error("No plot available. Please run the research & plot agent to generate a plot.")
    st.markdown('</div>', unsafe_allow_html=True)
 
 
with col2:
    st.markdown('<div class="info-container">', unsafe_allow_html=True)
    text_output_path = "output/info.txt"
    if os.path.exists(text_output_path):
        st.markdown("### Extracted Information:")
        with open(text_output_path, "r") as f:
            st.text_area("", f.read(), height=300)
    else:
        st.info("No additional information available.")
    st.markdown('</div>', unsafe_allow_html=True)
 
 
# Optional sidebar for extra controls or information
st.sidebar.title("Agent Dashboard Controls")
st.sidebar.info("This dashboard displays the output of the research & plot agent. Run the agent to generate a new plot and update the dashboard.")
 
 
# Footer
st.markdown('<div class="footer">Developed with using Streamlit</div>', unsafe_allow_html=True)
 
 
