import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Color Recognition Simulator | Abhishek Shaha",
    page_icon="🎨",
    layout="centered"
)

# -------------------------------
# Sidebar - Host & Project Info
# -------------------------------
with st.sidebar:
    st.header("👨‍💻 Project Host")
    st.markdown("""
    **Abhishek Shaha**  
    *Dept. of E&TC Engineering*  
    ---
    **About Simulator:**  
    Real-time RGB color extraction and virtual pump mixing simulation using Python & OpenCV.
    """)

# -------------------------------
# App Title & Host Banner
# -------------------------------
st.title("🎨 IoT-based Real-Time Color Recognition & Mixing Simulator")
st.caption("🚀 **Hosted & Developed by:** Abhishek Shaha")

st.markdown("""
This simulator extracts colors from an uploaded image, shows RGB values,  
and simulates the mixing process using virtual pumps.
""")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image using PIL and ensure 3 channels (RGB)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL image to OpenCV format (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # -------------------------------
    # Pixel Selection
    # -------------------------------
    st.subheader("Select Pixel Coordinates")

    max_x = img_cv.shape[1] - 1
    max_y = img_cv.shape[0] - 1

    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input(
            "X coordinate (width)",
            min_value=0,
            max_value=max_x,
            value=min(50, max_x // 2)
        )

    with col_y:
        y = st.number_input(
            "Y coordinate (height)",
            min_value=0,
            max_value=max_y,
            value=min(50, max_y // 2)
        )

    # Extract BGR values and convert to standard Python integers
    b, g, r = [int(val) for val in img_cv[int(y), int(x)]]
    rgb_color = (r, g, b)
    hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()

    st.write(f"**Extracted Color (R, G, B):** `{rgb_color}` | **HEX:** `{hex_color}`")

    # -------------------------------
    # Side-by-Side Color Comparison
    # -------------------------------
    st.markdown("### Color Comparison (Extracted vs. Mixed)")
    col1, col2 = st.columns(2)

    with col1:
        extracted_box = np.zeros((150, 150, 3), np.uint8)
        extracted_box[:] = (r, g, b)
        st.image(
            extracted_box,
            caption=f"Extracted Color\nRGB: ({r}, {g}, {b})",
            use_container_width=False
        )
        st.color_picker("Extracted Color Picker", hex_color)

    with col2:
        final_color = np.zeros((150, 150, 3), np.uint8)
        final_color[:] = (r, g, b)
        st.image(
            final_color,
            caption=f"Final Mixed Output\nRGB: ({r}, {g}, {b})",
            use_container_width=False
        )

    # -------------------------------
    # Pump Simulation Calculation
    # -------------------------------
    total = r + g + b if (r + g + b) > 0 else 1

    r_perc = (r / total) * 100
    g_perc = (g / total) * 100
    b_perc = (b / total) * 100

    st.markdown("### Pump Simulation (RGB Proportions)")
    p1, p2, p3 = st.columns(3)
    p1.metric("🔴 Red Pump", f"{r_perc:.2f}%")
    p2.metric("🟢 Green Pump", f"{g_perc:.2f}%")
    p3.metric("🔵 Blue Pump", f"{b_perc:.2f}%")

    # -------------------------------
    # Bar Chart Visualization
    # -------------------------------
    fig, ax = plt.subplots()
    pumps = ['Red', 'Green', 'Blue']
    values = [r_perc, g_perc, b_perc]

    ax.bar(pumps, values, color=['red', 'green', 'blue'])
    ax.set_ylabel("Pump Strength (%)")
    ax.set_ylim(0, 100)

    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2026 | Designed & Hosted by <b>Abhishek Shaha</b>"
    "</div>",
    unsafe_allow_html=True
)
