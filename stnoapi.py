import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model 
import time

# ----- Page Setup -----
st.set_page_config(page_title="AI for Circular Resource Intelligence",
                   page_icon="♻️", layout="centered")

st.markdown("""
<style>
body { background-color: #0e0e0e; color: #00FF00; }
.stButton>button { background-color: #00FF00; color: #000; }
</style>
""", unsafe_allow_html=True)

st.title("♻️ AI for Circular Resource Intelligence")
st.write("Upload a photo of any waste item to get AI-powered recycling guidance and local sustainability resources.")

# ----- Load Model -----
model = load_model("my_model.keras")
class_names = ["battery", "biological", "brown-glass", "cardboard", "clothes",
            "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]

recommendations_dict = {
    "battery": ["Keep used batteries in a sealed container.",
                    "Do not throw them in household trash.",
                    "Take them to an electronic or battery recycling point."],
    "biological": ["Compost organic waste like food scraps and leaves.",
                    "Avoid mixing biological waste with plastics.",
                    "Use closed compost bins to prevent odor."],
    "brown-glass": ["Rinse and separate from other glass colors.",
                    "Do not mix brown with clear or green glass.",
                    "Deliver to brown-glass recycling bins."],
    "cardboard": ["Flatten boxes before recycling to save space.",
                    "Keep them dry and free of grease or food stains.",
                    "Reuse for storage or crafts if clean."],
    "clothes": ["Donate wearable clothes to charity.",
                    "Repurpose old fabrics into cleaning rags.",
                    "Avoid burning textile waste."],
    "green-glass": ["Rinse and separate from clear or brown glass.",
                    "Reuse as vases or jars if safe.",
                    "Deliver to glass recycling centers."],
    "metal": ["Rinse cans and avoid mixing metals.",
                    "Flatten or crush large cans.",
                    "Donate scrap metal to recycling facilities."],
    "paper": ["Separate dry and wet paper.",
                    "Reuse for crafts, packaging, or notes.",
                    "Recycle if too torn or dirty."],
    "plastic": ["Clean and remove labels or caps.",
                    "Use for DIY crafts like planters.",
                    "Make eco-bricks with clean soft plastics."],
    "shoes": ["Donate wearable shoes to NGOs.",
                    "Repurpose old ones for planters or décor.",
                    "Recycle materials at textile collection points."],
    "trash": ["Avoid burning mixed trash.",
                    "Try to separate recyclable materials first.",
                    "Dispose of non-recyclables responsibly."],
    "white-glass": ["Rinse and separate from colored glass.",
                    "Handle with care to avoid breakage.",
                    "Recycle at glass drop-off stations."]
}

# ----- Country Selection -----
countries = ["Egypt", "UAE", "Kenya", "India", "Other"]
country = st.selectbox("Select Your Country:", countries)

# ----- File Uploader -----
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    progress_bar = st.progress(0, text="Analyzing Image...")
    time.sleep(1)
    progress_bar.progress(100)
    progress_bar.empty()

    # ----- Prediction -----
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    top_index = np.argmax(preds[0])
    detected_material = class_names[top_index]
    confidence = float(preds[0][top_index])

    st.subheader("🔍 Detected Material")
    st.write(f"**{detected_material.capitalize()}** ({confidence*100:.2f}% confidence)")

    st.subheader("♻️ Smart Recycling Steps")
    for i, step in enumerate(recommendations_dict.get(detected_material, ["No specific recommendations available."]), start=1):
        st.markdown(f"Step {i}: {step}")

    st.subheader("🌱 Local Recycling Resources")
    if country == "Egypt":
        st.markdown("""
        🇪🇬 **Egypt Eco Tips**
        - ♻️ Drop recyclables at [Go Green Initiative Centers](https://www.eeaa.gov.eg/en-us/topics/environmentaldevelopment/gogreeninitiative.aspx)
        - 💰 Exchange waste for rewards using [Bekia App](https://bekia.com.eg)
        - 🌿 Follow the [Eco Egypt Campaign](https://www.ecoegypt.org/)
        """)
    elif country == "UAE":
        st.markdown("""
        🇦🇪 **UAE Sustainability Programs**
        - 🗑️ Use [Recycle Right App](https://www.dm.gov.ae)
        - 🌱 Find drop-off points via [Bee’ah Recycling Platform](https://beeah.ae)
        - 🤝 Join [Emirates Environmental Group](https://www.eeg-uae.org/)
        """)
    elif country == "Kenya":
        st.markdown("""
        🇰🇪 **Kenya Green Solutions**
        - 🔄 Partner with [Mr. Green Africa](https://mrgreenafrica.com)
        - 🌾 Join organic composting programs by [Nairobi County](https://nairobi.go.ke)
        - 🏙️ Find community recycling stations in **Kilimani** and **Westlands**
        """)
    elif country == "India":
        st.markdown("""
        🇮🇳 **India Eco Actions**
        - 🧹 Deposit recyclables at [Swachh Bharat Kendra](https://swachhbharatmission.gov.in)
        - ♻️ Schedule pickups using [RecycleIndia App](https://play.google.com/store/apps/details?id=com.recycleindia)
        - 🌏 Join [plastic-free campaigns](https://www.plasticfreeindia.org)
        """)
    else:
        st.markdown("""
        🌎 **Global Tips**
        - 🔍 Find your nearest recycling center via [Earth911 Directory](https://earth911.com)
        - 🌿 Join local community sustainability groups
        - 💡 Try global apps like [TooGoodToGo](https://toogoodtogo.com/en) or [ShareWaste](https://sharewaste.com)
        """)

    if st.button("🗺️ Find Nearest Recycling Center"):
        map_query = f"https://www.google.com/maps/search/recycling+centers+in+{country.replace(' ', '+')}"
        st.markdown(f"[Open in Google Maps 🌍]({map_query})", unsafe_allow_html=True)

    st.caption("Last updated: October 2025")
