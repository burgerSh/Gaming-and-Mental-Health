import streamlit as st
import pandas as pd
import joblib
import altair as alt # เพิ่มไลบรารีนี้สำหรับวาดกราฟแนวนอนสวยๆ (ติดตั้งมาพร้อม Streamlit แล้ว)

# ==========================================
# 1. ตั้งค่าหน้าเว็บและการตกแต่ง (Custom CSS)
# ==========================================
st.set_page_config(page_title="Gaming Addiction Test", page_icon="🎮", layout="wide")
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/b/ba/Radioheadokcomputer.png", width=100)
    st.title("👨‍💻 เกี่ยวกับแอปนี้")
    st.write("แอปพลิเคชันนี้สร้างขึ้นเพื่อช่วยประเมินความเสี่ยงในการติดเกมเบื้องต้น ขับเคลื่อนด้วยโมเดล Machine Learning (Random Forest) ที่เรียนรู้จากชุดข้อมูลพฤติกรรมเกมเมอร์")
    st.markdown("---")
    st.write("📌 **พัฒนาโดย:** [นาย มัทธิว ขำดี]")
    st.write("📧 **ติดต่อ:** [ุ67160365@go.buu.ac.th]")
    st.markdown("---")
    st.caption("ข้อจำกัดความรับผิดชอบ: ผลลัพธ์จากการทำนายไม่ใช่การวินิจฉัยทางการแพทย์ หากคุณมีความกังวลเกี่ยวกับสุขภาพจิต โปรดปรึกษาแพทย์ผู้เชี่ยวชาญ")

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-text {
        font-size: 18px;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 30px;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. โหลดโมเดล
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load('gaming_risk_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    return model, model_cols

rf_model, model_columns = load_model()

st.markdown('<div class="main-title">🎮 ระบบ AI ประเมินความเสี่ยงการติดเกม</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">วิเคราะห์พฤติกรรมการเล่นเกมของคุณด้วย Machine Learning</div>', unsafe_allow_html=True)

# ==========================================
# 3. สร้าง Tabs เพื่อแยกหน้าการใช้งาน
# ==========================================
tab1, tab2 = st.tabs(["📝 ทำแบบประเมินความเสี่ยง", "🧠 เกี่ยวกับโมเดล AI"])

# ------------------------------------------
# TAB 1: หน้าแบบประเมิน (ฟอร์มเดิมของเรา)
# ------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("👤 **ข้อมูลส่วนตัว**")
        age = st.number_input("อายุ (ปี)", min_value=10, max_value=60, value=20)
        gender = st.selectbox("เพศ", ["Male", "Female", "Other"])
        social_iso = st.slider("คะแนนการแยกตัวจากสังคม (1=น้อย, 10=มาก)", 1, 10, 5)

    with col2:
        st.warning("🕹️ **ข้อมูลการเล่นเกม**")
        game_genre = st.selectbox("แนวเกมที่ชอบ", ["MOBA", "FPS", "RPG", "Mobile Games", "Battle Royale", "MMO"])
        platform = st.selectbox("แพลตฟอร์มหลัก", ["PC", "Console", "Mobile", "Multi-platform"])
        primary_game = st.selectbox("เกมที่เล่นประจำ (ตัวอย่าง)", ["Valorant", "Dota 2", "CS:GO", "Skyrim", "Cyberpunk 2077", "Clash of Clans", "Apex Legends", "อื่นๆ"])
        hours = st.slider("เล่นเกมกี่ชั่วโมง/วัน?", 0.0, 24.0, 3.0)
        spending = st.number_input("ค่าใช้จ่ายในเกมต่อเดือน (THB)", min_value=0, value=0)

    with col3:
        st.error("🩺 **อาการที่พบ (ติ๊กถูกหากมีอาการ)**")
        withdrawal = st.checkbox("หงุดหงิด/กระวนกระวายเมื่อไม่ได้เล่น")
        loss_interest = st.checkbox("เบื่อหน่ายกิจกรรมอื่นที่เคยชอบทำ")
        eye_strain = st.checkbox("ปวดตา / ตาล้าบ่อยครั้ง")
        back_pain = st.checkbox("ปวดหลัง / ปวดคอ")

    st.markdown("---")

    if st.button("🚀 คลิกเพื่อประเมินความเสี่ยง"):
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'daily_gaming_hours': [hours],
            'monthly_game_spending_usd': [spending],
            'game_genre': [game_genre],          
            'gaming_platform': [platform],       
            'withdrawal_symptoms': [1 if withdrawal else 0],
            'loss_of_other_interests': [1 if loss_interest else 0],
            'eye_strain': [1 if eye_strain else 0],
            'back_neck_pain': [1 if back_pain else 0],
            'social_isolation_score': [social_iso]
        })

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = rf_model.predict(input_encoded)[0]

        if prediction == "Low":
            st.success(f"### 🎉 ระดับความเสี่ยง: ต่ำ (Low)\nคุณมีการจัดการเวลาเล่นเกมได้ดีเยี่ยม! เล่นเพื่อความผ่อนคลายในระดับที่เหมาะสม")
            st.balloons()
        elif prediction == "Moderate":
            st.warning(f"### ⚠️ ระดับความเสี่ยง: ปานกลาง (Moderate)\nคุณเริ่มมีแนวโน้มการติดเกม ควรลองแบ่งเวลาไปทำกิจกรรมอื่นๆ หรือพบปะเพื่อนฝูงมากขึ้นนะครับ")
        else:
            st.error(f"### 🚨 ระดับความเสี่ยง: สูง (Severe)\nคุณมีความเสี่ยงสูงมากที่จะมีภาวะติดเกม ซึ่งอาจกระทบต่อสุขภาพและการใช้ชีวิต แนะนำให้ปรึกษาคนใกล้ชิดหรือผู้เชี่ยวชาญครับ")

# ------------------------------------------
# TAB 2: หน้าเกี่ยวกับโมเดล (ส่วนที่เพิ่มใหม่)
# ------------------------------------------
with tab2:
    st.header("🧠 เบื้องหลังการทำงานของ AI")
    st.write("""
    แอปพลิเคชันนี้ใช้โมเดล Machine Learning อัลกอริทึม **Random Forest Classifier** ซึ่งเรียนรู้รูปแบบจากชุดข้อมูลผู้เล่นเกมจำนวนมาก เพื่อค้นหาความสัมพันธ์ระหว่างพฤติกรรมการเล่นเกม สุขภาพจิต และความเสี่ยงในการติดเกม
    """)
    
    st.markdown("---")
    
    st.subheader("📊 ประสิทธิภาพของโมเดล (Model Performance)")
    st.write("ผลการทดสอบความแม่นยำของโมเดล (ประเมินจากชุดข้อมูลทดสอบ):")
    
    # สร้างกล่องตัวเลขโชว์ความแม่นยำ (อิงจากภาพที่คุณอัปโหลด)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="ความแม่นยำรวม (Accuracy)", value="88.44%", delta="เชื่อถือได้สูง")
    col_m2.metric(label="ความครอบคลุมกลุ่มเสี่ยงสูง (Recall)", value="87.00%", delta="สแกนเจอได้แม่นยำ")
    col_m3.metric(label="ค่าเฉลี่ยประสิทธิภาพ (F1-Score)", value="88.00%", delta="สมดุลดีเยี่ยม")

    st.markdown("---")

    st.subheader("📈 ปัจจัยหลักที่มีผลต่อการติดเกม (Top Feature Importance)")
    st.write("จากการวิเคราะห์ของ AI นี่คือตัวแปรที่มีน้ำหนักมากที่สุดในการตัดสินว่าใครมีความเสี่ยงในการติดเกม:")
    
    # สร้าง Dataframe จำลองจากภาพกราฟ Feature Importance ของคุณ
    feature_data = pd.DataFrame({
        "Feature": [
            "ชั่วโมงการเล่นเกมต่อวัน (Daily Hours)", 
            "ค่าใช้จ่ายในเกมต่อเดือน (Spending)", 
            "อายุ (Age)", 
            "การแยกตัวจากสังคม (Social Isolation)", 
            "อาการหงุดหงิดเมื่อไม่ได้เล่น (Withdrawal)", 
            "เบื่อหน่ายกิจกรรมอื่น (Loss of Interest)"
        ],
        "Importance": [0.28, 0.18, 0.16, 0.09, 0.04, 0.04]
    })
    
    # วาดกราฟแท่งแนวนอนด้วย Altair ให้สวยงาม
    chart = alt.Chart(feature_data).mark_bar(color='#FF4B4B', cornerRadiusEnd=4, height=30).encode(
        x=alt.X('Importance:Q', title='ระดับความสำคัญ (Importance Score)'),
        y=alt.Y('Feature:N', sort='-x', title='ปัจจัย (Features)'),
        tooltip=['Feature', 'Importance']
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.caption("หมายเหตุ: กราฟแสดงสัดส่วนน้ำหนักความสำคัญของแต่ละปัจจัยที่โมเดลใช้ในการตัดสินใจ (ค่าประมาณการจากโมเดล)")