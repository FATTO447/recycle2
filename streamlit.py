import streamlit as st
from PIL import Image
import io
import time
import requests

# ----- Page Setup -----

st.set_page_config(page_title="AI for Circular Resource Intelligence", page_icon="♻️", layout="centered")

st.markdown(""" <style>
body { background-color: #0e0e0e; color: #00FF00; }
.stButton>button { background-color: #00FF00; color: #000; } </style>
""", unsafe_allow_html=True)

st.title("♻️ AI for Circular Resource Intelligence")
st.write("Upload a photo of any waste item to get AI-powered recycling guidance and **local sustainability resources.**")

# ----- Country Selection -----

countries = ["Egypt", "UAE", "Kenya", "India", "Other"]
country = st.selectbox("Select Your Country:", countries)

# ----- File Uploader -----

uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # ----- Progress Bar -----  
    progress_bar = st.progress(0, text="Analyzing Image...")  
    time.sleep(1)  
    progress_bar.progress(100)  
    progress_bar.empty()  

    # ----- Convert Image to Bytes -----  
    img_bytes = io.BytesIO()  
    image.save(img_bytes, format="PNG")  
    img_bytes = img_bytes.getvalue()  

    api_url = "http://localhost:8000/predict"  
    try:  
        response = requests.post(api_url, files={"file": ("image.png", img_bytes, "image/png")})  
        if response.status_code == 200:  
            result = response.json()  
            top_preds = result.get("top_predictions", [])  

            st.subheader("🔍 Detected Material Predictions")  
            if top_preds:  
                for pred in top_preds:  
                    label = pred["label"].capitalize()  
                    confidence = pred["confidence"]  
                    # Correct confidence percentage  
                    confidence_percentage = confidence if confidence > 1 else confidence * 100  
                    st.write(f"**{label}** ({confidence_percentage:.2f}% confidence)")  

                    recommendations = pred.get("recommendations", ["No specific recommendations available."])  
                    st.markdown("♻️ **Smart Recycling Steps:**")  
                    for i, step in enumerate(recommendations, start=1):  
                        st.markdown(f"Step {i}: {step}")  
                    st.markdown("---")  
            else:  
                st.write("No predictions returned.")  

            # ----- Local Recycling Resources -----  
            st.subheader("🌱 Local Recycling Resources")  
            if country == "Egypt":  
                st.markdown("""  
                🇪🇬 **Egypt Eco Tips**  
                - ♻️ Drop recyclables at [Go Green Initiative Centers](https://www.eeaa.gov.eg/en-us/topics/environmentaldevelopment/gogreeninitiative.aspx).  
                - 💰 Exchange waste for rewards using [Bekia App](https://bekia.com.eg).  
                - 🌿 Follow the [Eco Egypt Campaign](https://www.ecoegypt.org/).  
                """)  
            elif country == "UAE":  
                st.markdown("""  
                🇦🇪 **UAE Sustainability Programs**  
                - 🗑️ Use [Recycle Right App](https://www.dm.gov.ae).  
                - 🌱 Find drop-off points via [Bee’ah Recycling Platform](https://beeah.ae).  
                - 🤝 Join [Emirates Environmental Group](https://www.eeg-uae.org/).  
                """)  
            elif country == "Kenya":  
                st.markdown("""  
                🇰🇪 **Kenya Green Solutions**  
                - 🔄 Partner with [Mr. Green Africa](https://mrgreenafrica.com).  
                - 🌾 Join organic composting programs by [Nairobi County](https://nairobi.go.ke).  
                - 🏙️ Find community recycling stations in **Kilimani** and **Westlands**.  
                """)  
            elif country == "India":  
                st.markdown("""  
                🇮🇳 **India Eco Actions**  
                - 🧹 Deposit recyclables at [Swachh Bharat Kendra](https://swachhbharatmission.gov.in/).  
                - ♻️ Schedule pickups using [RecycleIndia App](https://play.google.com/store/apps/details?id=com.recycleindia).  
                - 🌏 Join [plastic-free campaigns](https://www.plasticfreeindia.org/).  
                """)  
            else:  
                st.markdown("""  
                🌎 **Global Tips**  
                - 🔍 Find your nearest recycling center via [Earth911 Directory](https://earth911.com/).  
                - 🌿 Join local community sustainability groups.  
                - 💡 Try global apps like [TooGoodToGo](https://toogoodtogo.com/en) or [ShareWaste](https://sharewaste.com/).  
                """)  

            # ----- Google Maps Link -----  
            if st.button("🗺️ Find Nearest Recycling Center"):  
                map_query = f"https://www.google.com/maps/search/recycling+centers+in+{country.replace(' ', '+')}"  
                st.markdown(f"[Open in Google Maps 🌍]({map_query})", unsafe_allow_html=True)  

            st.caption("Last updated: October 2025")  

        else:  
            st.error(f"⚠️ API error — received {response.status_code}.")  
    except Exception as e:  
        st.error(f"Connection error: {e}")  
