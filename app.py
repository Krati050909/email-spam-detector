import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import pipeline

@st.cache_resource
def load_all_tools():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        nltk.download('stopwords', quiet=True)
        return model, vectorizer, pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"❌ System Initialization Error: {e}")
        return None, None, None

model, vectorizer, ai_classifier = load_all_tools()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

def extract_urls(text):
    return re.findall(r'(https?://\S+)', text)

st.set_page_config(page_title="Phishing Shield v8.0", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .medium-font { font-size:18px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Phishing Shield: Contextual Intelligence")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("📩 Email Analysis:", height=250, placeholder="Paste email content here...")
    uploaded_file = st.file_uploader("Or upload .txt", type=["txt"])
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")

with col2:
    st.markdown("### ⚙️ Analysis Settings")
    use_ai = st.toggle("🤖 Enable AI Linguistic Scan", value=True)
    st.info("💡 This version handles 'Get-Rich-Quick' scams, OTP requests, and Urgency traps even without links.")

if st.button("🚀 EXECUTE FULL SCAN", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please provide email text.")
    elif model is None:
        st.error("Engines Offline.")
    else:

        msg_clean = clean_text(user_input)
        msg_vec = vectorizer.transform([msg_clean])
        prob = model.predict_proba(msg_vec)[0]
        base_risk = prob[1] 

        label, conf = "POSITIVE", 0.0
        if use_ai:
            with st.spinner("🤖 AI evaluating intent..."):
                ai_result = ai_classifier(user_input[:512])[0]
                label, conf = ai_result["label"], ai_result["score"] * 100


        urls = extract_urls(user_input)
        risk_adjustment = 0.0
        

        fraud_patterns = ["otp", "bank details", "won ₹", "winner", "claim prize", "lucky winner"]
        urgency_patterns = ["suspension", "24 hours", "act fast", "immediately", "account closure"]
        job_scam_keywords = ["earn", "daily", "salary", "no skills", "limited slots", "income", "profit", "work from home"]
        
        fraud_hits = sum(1 for p in fraud_patterns if p in user_input.lower())
        urgency_hits = sum(1 for p in urgency_patterns if p in user_input.lower())
        job_scam_hits = sum(1 for w in job_scam_keywords if w in user_input.lower())


        super_trusted = ["amazon.in", "amazon.com", "google.com", "microsoft.com", "apple.com", "sbi.co.in", "hdfcbank.com"]
        
        is_link_super_safe = False
        if urls:
            for url in urls:
                domain = url.split("//")[-1].split("/")[0].lower()
                if any(t in domain for t in super_trusted):
                    is_link_super_safe = True

        if fraud_hits >= 2 or job_scam_hits >= 3:

            risk_adjustment = 1.0 
        elif is_link_super_safe:

            risk_adjustment -= 0.8
        elif not urls:

            if fraud_hits == 0 and job_scam_hits < 2 and urgency_hits == 0:
                risk_adjustment -= 0.6 
                if label == "POSITIVE": risk_adjustment -= 0.2
            else:

                risk_adjustment += 0.5 
        else:

            if fraud_hits > 0 or job_scam_hits > 1 or urgency_hits > 0:
                risk_adjustment += 0.5
            for url in urls:
                if url.startswith("http://"): risk_adjustment += 0.3

        final_risk = max(0.0, min(base_risk + risk_adjustment, 1.0))

        st.markdown("## 📊 Diagnostic Report")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("⚡ Threat Assessment")
            if final_risk > 0.60:
                st.error(f"### 🚨 PHISHING / SCAM DETECTED")
            elif final_risk > 0.30:
                st.warning(f"### ⚠️ SUSPICIOUS ACTIVITY")
            else:
                st.success(f"### ✅ APPARENTLY SAFE")
            
            st.write(f"**Final Risk Score:** `{final_risk*100:.1f}%`")
            st.progress(float(final_risk))

        with res_col2:
            st.subheader("🔗 Link Scan")
            if urls:
                for url in urls:
                    is_safe = any(t in url.lower() for t in super_trusted)
                    st.write(f"{'✅ Trusted' if is_safe else '❓ Unknown'}: `{url}`")
            else:
                st.write("No external links detected. (Contextual Trust Applied)")

        if use_ai:
            st.markdown("---")
            is_bad = (final_risk > 0.5)
            bg_color = "#f8d7da" if is_bad else "#d4edda"
            text_color = "#721c24" if is_bad else "#155724"
            st.markdown(f"""
                <div style="background-color:{bg_color}; padding:25px; border-radius:15px; border-left: 10px solid {text_color};">
                    <p style="color:{text_color}; margin-bottom:0;" class="medium-font">🤖 <b>AI Verdict: {'DANGEROUS' if is_bad else 'SAFE'}</b></p>
                    <h2 style="color:{text_color}; margin-top:0;" class="big-font">{'SCAM DETECTED' if is_bad else 'NORMAL COMMUNICATION'} ({conf:.1f}% Match)</h2>
                    <p style="color:{text_color};">Analysis: Found {fraud_hits} financial markers and {job_scam_hits} job scam markers. Tone is {label.lower()}.</p>
                </div>
            """, unsafe_allow_html=True)
