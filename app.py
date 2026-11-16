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
# Automatic Lighting & Shadow Correction Function
# -------------------------------
def auto_correct_lighting(img_bgr):
    # Gray-World White Balance Algorithm (अंधार आणि पिवळसर लाईटचा परिणाम आपोआप काढण्यासाठी)
    result = img_bgr.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / max(1.0, avg_b)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / max(1.0, avg_g)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / max(1.0, avg_r)), 0, 255)
    balanced = result.astype(np.uint8)

    # LAB Color Space मध्ये Brightness (L channel) संतुलित करणे
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    corrected_bgr = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    return corrected_bgr

# -------------------------------
# Closest Standard Shade Matcher
# -------------------------------
STANDARD_SHADES = {
    "Crimson Dark Red": (139, 0, 0),
    "Signal Red": (200, 16, 46),
    "Terracotta / Brick Red": (165, 42, 42),
    "Sunset Orange": (255, 94, 19),
    "Marigold Yellow": (255, 195, 0),
    "Emerald Green": (0, 155, 119),
    "Olive Leaf Green": (85, 107, 47),
    "Royal Navy Blue": (0, 35, 102),
    "Sky Cerulean Blue": (0, 123, 167),
    "Lavender Mist": (181, 126, 220),
    "Charcoal Black": (34, 34, 34),
    "Warm Ivory White": (245, 245, 220)
}

def find_closest_shade(r, g, b):
    best_name = "Custom Shade"
    min_dist = float("inf")
    for name, (sr, sg, sb) in STANDARD_SHADES.items():
        dist = np.sqrt((r - sr)**2 + (g - sg)**2 + (b - sb)**2)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    # Accuracy percentage (Max possible RGB distance is ~441.67)
    accuracy = max(0.0, (1 - (min_dist / 441.67)) * 100)
    return best_name, accuracy

# -------------------------------
# Sidebar - Host Info
# -------------------------------
with st.sidebar:
    st.header("👨‍💻 Project Host")
    st.markdown("""
    **Abhishek Shaha**  
    *Dept. of E&TC Engineering*  
    ---
    **Smart Features Enabled:**  
    ✅ Auto Low-Light & Shadow Fix  
    ✅ Auto Center-Patch Sampling  
    ✅ Direct Copyable HEX & Shade Match
    """)

# -------------------------------
# App Title & Host Banner
# -------------------------------
st.title("🎨 IoT-based Real-Time Color Recognition & Mixing Simulator")
st.caption("🚀 **Hosted & Developed by:** Abhishek Shaha")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload or Capture Color Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_bgr_raw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # युजरला त्रास न देता आपोआप लाइट दुरुस्त करणे
    auto_fix = st.toggle("✨ Auto Low-Light & Shadow Correction", value=True)
    img_cv = auto_correct_lighting(img_bgr_raw) if auto_fix else img_bgr_raw

    # इमेजचा मध्यभाग आपोआप निवडणे (हवे असल्यास युजर बदलू शकतो)
    h, w, _ = img_cv.shape
    default_x, default_y = w // 2, h // 2

    with st.expander("🎯 Adjust Target Point (Optional - Auto-centered by default)"):
        col_x, col_y = st.columns(2)
        with col_x:
            x = st.slider("X Position", 0, w - 1, default_x)
        with col_y:
            y = st.slider("Y Position", 0, h - 1, default_y)

    # निवडलेल्या भागाभोवती 15x15 सरासरी रंग घेणे (कॅमेरा नॉइज टाळण्यासाठी)
    patch = img_cv[max(0, y-7):min(h, y+8), max(0, x-7):min(w, x+8)]
    avg_b, avg_g, avg_r = [int(val) for val in np.mean(patch, axis=(0, 1))]
    r, g, b = avg_r, avg_g, avg_b

    hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
    shade_name, match_acc = find_closest_shade(r, g, b)

    # इमेजवर टार्गेट पॉइंटर दाखवणे
    preview_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).copy()
    cv2.circle(preview_img, (x, y), max(6, min(h, w) // 25), (255, 255, 255), 2)
    st.image(preview_img, caption="Calibrated Image (White circle shows sampled area)", use_container_width=True)

    # -------------------------------
    # Direct HEX Code & Shade Match Output (शोधत बसायची गरज नाही)
    # -------------------------------
    st.markdown("### 🎯 Direct Result (Ready for Shop / Mixing)")
    res1, res2, res3 = st.columns(3)
    with res1:
        st.markdown("**Copy HEX Code:**")
        st.code(hex_color, language=None)
    with res2:
        st.metric("Nearest Shade", shade_name)
    with res3:
        st.metric("Shop Match Confidence", f"{match_acc:.1f}%")

    # -------------------------------
    # Side-by-Side Color Comparison
    # -------------------------------
    st.markdown("### Color Comparison (Calibrated vs. Mixed)")
    col1, col2 = st.columns(2)

    with col1:
        extracted_box = np.zeros((150, 150, 3), np.uint8)
        extracted_box[:] = (r, g, b)
        st.image(extracted_box, caption=f"Calibrated Color\nRGB: ({r}, {g}, {b})", use_container_width=False)

    with col2:
        final_color = np.zeros((150, 150, 3), np.uint8)
        final_color[:] = (r, g, b)
        st.image(final_color, caption=f"Final Mixed Output\nHEX: {hex_color}", use_container_width=False)

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
    ax.bar(['Red', 'Green', 'Blue'], [r_perc, g_perc, b_perc], color=['red', 'green', 'blue'])
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
