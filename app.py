import streamlit as st
import pandas as pd
import joblib
import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image
import shutil

# --- CONFIGURATION & MODEL LOADING ---
RESPONSES_CSV = "survey_responses_gradio.csv"
IMAGE_SAVE_DIR = "captured_images"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- Load Models (Cached for performance) ---
@st.cache_resource
def load_survey_model():
    try:
        return joblib.load("random_forest_model_corrected.joblib")
    except Exception as e:
        st.sidebar.error(f"FATAL ERROR loading survey model: {e}")
        return None

@st.cache_resource
def load_image_model():
    try:
        return tf.keras.models.load_model("facial_emotion_model.h5")
    except Exception as e:
        st.sidebar.error(f"FATAL ERROR loading image model: {e}")
        return None

survey_model = load_survey_model()
image_model = load_image_model()

if survey_model:
    st.sidebar.success("✅ Survey model loaded.")
if image_model:
    st.sidebar.success("✅ Image emotion model loaded.")

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprize']
IMG_HEIGHT, IMG_WIDTH = 48, 48

# --- HELPER DATA ---
dass_mapping = {
    "i found it hard to wind down": "Q3_1_S1", "I tended to over-react to situations.": "Q3_2_S2",
    "I felt that I was using a lot of nervous energy.": "Q3_3_S3", "I found myself getting agitated.": "Q3_4_S4",
    "I found it difficult to relax.": "Q3_5_S5", "I was intolerant of anything that kept me from getting on with what I was doing.": "Q3_6_S6",
    "I felt that I was rather touchy.": "Q3_7_S7", "I was aware of dryness of my mouth.": "Q3_8_A1",
    "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness without physical exertion).": "Q3_9_A2",
    "I experienced trembling (e.g., in the hands).": "Q3_10_A3", "I was worried about situations in which I might panic and make a fool of myself.": "Q3_11_A4",
    "I felt I was close to panic.": "Q3_12_A5", "I was aware of the action of my heart without physical exertion (e.g., heart rate increase).": "Q3_13_A6",
    "I felt scared without any good reason.": "Q3_14_A7", "I couldn’t seem to experience any positive feeling at all": "Q3_15_D1",
    "I found it difficult to work up the initiative to do things.": "Q3_16_D2", "I felt that I had nothing to look forward to.": "Q3_17_D3",
    "I felt down-hearted and blue.": "Q3_18_D4", "I was unable to become enthusiastic about anything.": "Q3_19_D5",
    "I felt I wasn’t worth much as a person.": "Q3_20_D6", "I felt that life was meaningless.": "Q3_21_D7"
}
questions_text = list(dass_mapping.keys())
response_options = [
    "0: Did not apply to me at all",
    "1: Applied to me to some degree, or some of the time",
    "2: Applied to me to a considerable degree, or a good part of time",
    "3: Applied to me very much, or most of the time"
]
# Feature columns the CORRECTED survey model expects (from your notebook)
feature_cols_survey_model = [
    'Q3_1_S1', 'Q3_2_S2', 'Q3_3_S3', 'Q3_4_S4', 'Q3_5_S5', 'Q3_6_S6', 'Q3_7_S7',
    'Q3_8_A1', 'Q3_9_A2', 'Q3_10_A3', 'Q3_11_A4', 'Q3_12_A5', 'Q3_13_A6', 'Q3_14_A7',
    'Q3_15_D1', 'Q3_16_D2', 'Q3_17_D3', 'Q3_18_D4', 'Q3_19_D5', 'Q3_20_D6', 'Q3_21_D7',
    'Stress_Score', 'Anxiety_Score'
]

# --- CALCULATE SCORES ---
def calculate_scores(responses_coded):
    stress_cols = [f"Q3_{i}_S{j}" for i, j in zip(range(1, 8), range(1, 8))]
    anxiety_cols = [f"Q3_{i}_A{j}" for i, j in zip(range(8, 15), range(1, 8))]
    depression_cols = [f"Q3_{i}_D{j}" for i, j in zip(range(15, 22), range(1, 8))]
    scores = {
        "Stress_Score": sum(responses_coded.get(col, 0) for col in stress_cols),
        "Anxiety_Score": sum(responses_coded.get(col, 0) for col in anxiety_cols),
        "Depression_Score": sum(responses_coded.get(col, 0) for col in depression_cols)
    }
    return scores

# --- IMAGE ANALYSIS ---
def analyze_image(image_input):
    if image_input is None or image_model is None:
        return "No Image Provided"
    try:
        # Convert PIL Image to NumPy array
        image = np.array(image_input)
        # Convert to grayscale for the model
        gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray_frame, (IMG_WIDTH, IMG_HEIGHT))
        img_batch = np.expand_dims(resized, axis=(0, -1))
        
        predictions = image_model.predict(img_batch, verbose=0)
        return EMOTION_CLASSES[np.argmax(predictions[0])]
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"Error analyzing image: {e}"

# --- *** ENHANCED SUGGESTIONS FUNCTION *** ---
def get_suggestions(survey_prediction, image_emotion, scores):
    suggestions = ""
    is_likely_depressed = (survey_prediction == "Likely Depressed")
    is_high_risk = (is_likely_depressed or image_emotion in ['Sad', 'Fear', 'Angry'] or scores.get("Depression_Score", 0) > 13)

    if is_likely_depressed:
        suggestions += "**Based on your survey responses, your results show signs consistent with depression.**\n\n"
    elif is_high_risk:
        suggestions += f"**Based on the image analysis (detecting {image_emotion}) and your scores, it's important to be mindful of your well-being.**\n\n"
    else:
        st.markdown(f"""
        **Based on your screening, you do not currently show significant signs of depression.**  
        * (Image emotion detected: {image_emotion})  
        * Continue to practice self-care and monitor your mood.  
        """)
        return

    st.markdown(f"""
    **This is a screening tool, not a diagnosis.** Please consider these helpful steps:

    * **Seek Professional Help:** This is the most effective step you can take. Schedule an appointment with your doctor 👨‍⚕️, a therapist 🧠, counselor, or a mental health professional. They can provide an accurate diagnosis if needed and discuss the best treatment options for you, which might include therapy, medication, or lifestyle adjustments. Don't hesitate – reaching out is a sign of strength. 🩺

    * **Talk to Someone You Trust:** Sharing your feelings can make a significant difference. Reach out to a close friend 🧑‍🤝‍🧑, family member 👨‍👩‍👧‍👦, partner ❤️, or anyone you feel comfortable confiding in. You don't have to go through this alone. 🤗

    * **Focus on Self-Care Basics:**  
      * **Sleep:** Aim for a consistent sleep schedule (7–9 hours). 😴🌙  
      * **Nutrition:** Eat regular, balanced meals. 🍎🥦  
      * **Movement:** Even a short daily walk outdoors can help. 🚶‍♀️☀️  

    * **Practice Mindfulness & Relaxation:** Try deep breathing 😮‍💨, guided meditations, or gentle yoga 🧘 to calm your mind.

    * **Engage (Gently) in Activities:** Try to re-engage with small enjoyable activities like listening to music 🎶, reading 📚, or drawing 🎨.

    * **Limit Alcohol & Substances:** These can worsen depression symptoms. 🚫🍺

    * **Be Patient and Kind to Yourself:** Healing takes time; celebrate small victories and avoid self-criticism. ❤️⏳

    ---

    **Remember:** You are not alone — many people experience these feelings, and effective help is available. Taking the first step to reach out is often the hardest but most important one. 💪🫂

    **Helpline Numbers (India):**  
    * **AASRA:** +91-9820466726 (24×7)  
    * **Vandrevala Foundation:** +91-9999-666-555 (24×7)
    """)
    return suggestions

# --- SAVE RESULTS ---
def save_results(name, image_input, survey_responses_dict, scores, survey_pred, image_emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_path = "N/A"

    if image_input is not None:
        try:
            # Convert PIL Image to NumPy array (RGB)
            image_np = np.array(image_input)
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            image_filename = f"{safe_name}_{int(datetime.now().timestamp())}.png"
            image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
            
            cv2.imwrite(image_path, image_bgr)
            print(f"Image saved to: {image_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            image_path = "Error saving image"

    data = {
        "Timestamp": timestamp,
        "Name": name,
        "Image_Filename": image_path,
        **scores,
        "Survey_Prediction": survey_pred,
        "Image_Emotion": image_emotion,
        **survey_responses_dict
    }
    df_new = pd.DataFrame([data])
    df_new.to_csv(RESPONSES_CSV, mode='a', header=not os.path.exists(RESPONSES_CSV), index=False)

# --- MAIN PAGE UI ---
st.set_page_config(layout="wide")
st.title("🧠 DASS-21 & Image Emotion Screening Tool")
st.markdown("This tool screens for signs of depression. **It is not a medical diagnosis.**")

# Check if models loaded
if survey_model is None or image_model is None:
    st.error("A critical model failed to load. Please check the logs and restart the app.")
else:
    tab1, tab2 = st.tabs(["🩺 Start Screening", "📊 View My Results"])

    # --- TAB 1: START SCREENING ---
    with tab1:
        st.subheader("Step 1: Consent")
        consent = st.checkbox("I consent and confirm I am over 18 years old. I understand this is not a diagnosis and my data (name, responses, and image) will be saved.", key="consent_check")
        st.markdown("---")

        if consent:
            name = st.text_input("Step 2: Enter your Name (or a unique ID)*", key="name_input")
            
            st.subheader("Step 3: Capture Snapshot")
            st.write("Please look directly at the camera. This image will be analyzed for a dominant emotion.")
            image_buffer = st.camera_input("Take a picture 📸", key="camera")
            
            st.markdown("---")
            
            st.subheader("Step 4: Questionnaire")
            st.write("Please read each statement and select the option which indicates how much it applied to you **over the past week**.")
            
            survey_responses = {}
            with st.expander("Click to show DASS-21 Questions", expanded=True):
                # Create 2 columns for a cleaner layout
                col1, col2 = st.columns(2)
                for i, q_text in enumerate(questions_text):
                    # Alternate questions between the two columns
                    if i % 2 == 0:
                        with col1:
                            selected_option = st.radio(f"{i+1}. {q_text}", options=response_options, key=f"q_{i}", index=None)
                    else:
                        with col2:
                            selected_option = st.radio(f"{i+1}. {q_text}", options=response_options, key=f"q_{i}", index=None)
                    
                    survey_responses[q_text] = int(selected_option.split(":")[0]) if selected_option else None

            st.markdown("---")
            
            if st.button("Submit & Analyze Results", type="primary"):
                # --- Validation ---
                all_answered = all(v is not None for v in survey_responses.values())
                
                if not name:
                    st.error("⚠️ Please enter your name.")
                elif image_buffer is None:
                    st.error("⚠️ Please take a snapshot.")
                elif not all_answered:
                    st.error("⚠️ Please answer all 21 questions.")
                else:
                    with st.spinner("Analyzing your results..."):
                        # --- 1. Process Image ---
                        pil_image = Image.open(image_buffer)
                        emotion = analyze_image(pil_image)

                        # --- 2. Process Survey ---
                        responses_coded = {dass_mapping[q]: v for q, v in survey_responses.items()}
                        scores = calculate_scores(responses_coded)
                        
                        input_data_dict = {
                            **responses_coded,
                            "Stress_Score": scores["Stress_Score"],
                            "Anxiety_Score": scores["Anxiety_Score"]
                        }
                        
                        input_df = pd.DataFrame([input_data_dict])
                        input_df_ordered = input_df[feature_cols_survey_model]
                        
                        # Note: The 'cell6' model was trained WITHOUT scaling, so we predict directly
                        pred_val = survey_model.predict(input_df_ordered)[0]
                        survey_label = "Likely Depressed" if pred_val == 1 else "Not Depressed"
                        
                        # --- 3. Save & Display ---
                        save_results(name, pil_image, responses_coded, scores, survey_label, emotion)
                        
                        st.success("✅ Submission Successful! See your results below.")
                        
                        st.subheader("Your Screening Results")
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Stress Score", scores['Stress_Score'])
                            st.metric("Anxiety Score", scores['Anxiety_Score'])
                            st.metric("Depression Score", scores['Depression_Score'])
                        with res_col2:
                            st.metric("Survey Prediction", survey_label)
                            st.metric("Detected Emotion", emotion)
                        
                        st.markdown("---")
                        st.subheader("Suggestions & Next Steps")
                        st.markdown(get_suggestions(survey_label, emotion, scores))
        else:
            st.info("Please check the consent box to begin the screening.")

    # --- TAB 2: VIEW RESULTS ---
    with tab2:
        st.subheader("Retrieve Your Results")
        search_name = st.text_input("Enter your Name or ID", key="search_name")
        if st.button("Load My Results", key="load_results"):
            if not os.path.exists(RESPONSES_CSV):
                st.error("No records found.")
            elif not search_name:
                st.error("Please enter a name or ID to search.")
            else:
                with st.spinner("Searching..."):
                    df = pd.read_csv(RESPONSES_CSV)
                    # Search for the most recent record matching the name
                    row = df[df["Name"] == search_name].tail(1)
                    
                    if row.empty:
                        st.error(f"No results found for '{search_name}'.")
                    else:
                        row = row.iloc[0]
                        st.success(f"Displaying the most recent record for **{row['Name']}** (from {row['Timestamp']})")
                        
                        st.subheader("Stored Screening Results")
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Stress Score", row['Stress_Score'])
                            st.metric("Anxiety Score", row['Anxiety_Score'])
                            st.metric("Depression Score", row['Depression_Score'])
                        with res_col2:
                            st.metric("Survey Prediction", row['Survey_Prediction'])
                            st.metric("Detected Emotion", row['Image_Emotion'])
                        
                        st.markdown("---")
                        st.subheader("Suggestions & Next Steps")
                        scores_dict = {
                            "Stress_Score": row['Stress_Score'],
                            "Anxiety_Score": row['Anxiety_Score'],
                            "Depression_Score": row['Depression_Score']
                        }
                        st.markdown(get_suggestions(row['Survey_Prediction'], row['Image_Emotion'], scores_dict))
                        
                        # Display the saved image
                        image_path = row.get("Image_Filename")
                        if image_path and os.path.exists(image_path):
                            st.subheader("Snapshot on File")
                            st.image(image_path, caption="The snapshot taken during your screening.")
                        else:
                            st.warning("Could not find the snapshot file for this record.")