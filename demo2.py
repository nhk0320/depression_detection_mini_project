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
from supabase import create_client, Client  # --- ADDED ---

# --- CONFIGURATION & MODEL LOADING ---
RESPONSES_CSV = "survey_responses_streamlit.csv"
IMAGE_SAVE_DIR = "captured_images_streamlit"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- Load Models (Cached) ---
@st.cache_resource
def load_survey_model():
    try: return joblib.load("random_forest_model_corrected.joblib")
    except Exception as e: st.sidebar.error(f"Survey model load error: {e}"); return None

@st.cache_resource
def load_image_model():
    try: return tf.keras.models.load_model("facial_emotion_model.h5")
    except Exception as e: st.sidebar.error(f"Image model load error: {e}"); return None

@st.cache_resource
def load_eeg_model():
    # Attempt to load the EEG model file provided by the user
    try: return joblib.load("eeg_model.pkl") # Use the uploaded filename
    except Exception as e: st.sidebar.warning(f"EEG model (eeg_model.pkl) not found."); return None

# --- ADDED: Supabase Connection ---
@st.cache_resource
def init_connection():
    """Initializes the Supabase connection using Streamlit Secrets."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.sidebar.error(f"Supabase connection error: {e}")
        return None
# --- END ADDED ---

survey_model = load_survey_model()
image_model = load_image_model()
eeg_model = load_eeg_model() # Load the new model
supabase: Client = init_connection() # --- ADDED ---

# Sidebar status
st.sidebar.title("Model Status")
if survey_model: st.sidebar.success("✅ Survey model loaded.")
if image_model: st.sidebar.success("✅ Image model loaded.")
if eeg_model: st.sidebar.success("✅ EEG model loaded (Simulation).")
else: st.sidebar.warning("⚠️ EEG model not found.")
if supabase: st.sidebar.success("✅ Database connected.") # --- ADDED ---


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
feature_cols_survey_model = [
    'Q3_1_S1', 'Q3_2_S2', 'Q3_3_S3', 'Q3_4_S4', 'Q3_5_S5', 'Q3_6_S6', 'Q3_7_S7',
    'Q3_8_A1', 'Q3_9_A2', 'Q3_10_A3', 'Q3_11_A4', 'Q3_12_A5', 'Q3_13_A6', 'Q3_14_A7',
    'Q3_15_D1', 'Q3_16_D2', 'Q3_17_D3', 'Q3_18_D4', 'Q3_19_D5', 'Q3_20_D6', 'Q3_21_D7',
    'Stress_Score', 'Anxiety_Score'
]

# --- HELPER FUNCTIONS ---
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

def analyze_image(image_input):
    if image_input is None or image_model is None: return "N/A"
    try:
        image = np.array(image_input)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        img_batch = np.expand_dims(resized, axis=(0, -1))
        pred = image_model.predict(img_batch, verbose=0)
        return EMOTION_CLASSES[np.argmax(pred[0])]
    except Exception as e: print(f"Img err: {e}"); return "Error"

def get_suggestions(survey_pred, image_emo, scores, eeg_pred):
    # (Same enhanced function as before)
    is_likely_depressed = (survey_pred == "Likely Depressed") or (eeg_pred == "Depressed (Simulated)")
    is_high_risk = (is_likely_depressed or image_emo in ['Sad','Fear','Angry'] or scores.get("Depression_Score", 0) > 13)

    if not is_likely_depressed and not is_high_risk:
        st.success(f"""
        **Based on your screening, you do not currently show significant signs of depression.**
        * (Image emotion detected: {image_emo})
        * (EEG status: {eeg_pred})
        * Continue to practice self-care and monitor your mood.
        """)
        return

    if is_likely_depressed:
        st.warning("#### Based on your survey and/or EEG results, your screening shows signs consistent with depression.")
    elif is_high_risk:
        st.warning(f"#### Based on the analysis (detecting **{image_emo}**), it's important to be mindful of your well-being.")

    st.markdown("""
    **This is a screening tool, not a diagnosis.** Please consider these helpful steps:

    * **Seek Professional Help:** This is the most effective step. Talk to a doctor 👨‍⚕️, therapist 🧠, or mental health professional. 🩺
    * **Talk to Someone You Trust:** Share your feelings with a friend 🧑‍🤝‍🧑, family 👨‍👩‍👧‍👦, or partner ❤️. 🤗
    * **Focus on Self-Care Basics:** Maintain routines for sleep 😴🌙, nutrition 🍎🥦, and gentle physical activity 🚶‍♀️☀️.
    * **Practice Mindfulness & Relaxation:** Try deep breathing 😮‍💨, meditation, or gentle yoga 🧘.
    * **Engage (Gently) in Activities:** Reconnect with hobbies or simple pleasures 🎶🌳📚🎨.
    * **Limit Alcohol & Substances:** These can worsen symptoms. 🚫🍺
    * **Be Patient and Kind to Yourself:** Healing takes time. ❤️⏳

    ---
    ### **If you are in crisis or feel you are a danger to yourself, please reach out for help immediately.**
    **You are not alone, and help is available.** 💪🫂

    * **AASRA (India):** +91-9820466726 (24/7 Helpline)
    * **Vandrevala Foundation (India):** +91 9999 666 555 (24/7 Helpline)
    """)
    st.error("**Crucially, please consult a healthcare professional for a proper evaluation and guidance.**")

def save_results(name, image_input, survey_responses_dict, scores, survey_pred, image_emo, eeg_pred):
    # --- 1. Your existing local saving code (unchanged) ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_path = "N/A"
    if image_input is not None:
        try:
            image_np = np.array(image_input)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            img_filename = f"{safe_name}_{int(datetime.now().timestamp())}.png"
            image_path = os.path.join(IMAGE_SAVE_DIR, img_filename)
            cv2.imwrite(image_path, image_bgr)
            print(f"Image saved: {image_path}")
        except Exception as e: print(f"Img save err: {e}"); image_path = "Error"
    
    data = {
        "Timestamp": timestamp, "Name": name, "Image_Filename": image_path,
        **scores, "Survey_Prediction": survey_pred, "Image_Emotion": image_emo,
        "EEG_Prediction": eeg_pred, **survey_responses_dict
    }
    df = pd.DataFrame([data])
    df.to_csv(RESPONSES_CSV, mode='a', header=not os.path.exists(RESPONSES_CSV), index=False)
    print(f"Data saved for {name} to {RESPONSES_CSV}")

    # --- 2. ADDED: Save results to Supabase ---
    if supabase:
        try:
            # Create a dictionary for the database
            # (Use lowercase and underscores for table column names)
            db_data = {
                "username": name,
                "image_emotion": image_emo if image_emo != "Error" else None,
                "survey_prediction": survey_pred if survey_pred != "Error" else None,
                "eeg_prediction": eeg_pred if eeg_pred not in ["N/A", "Error"] else None,
                "stress_score": scores.get("Stress_Score"),
                "anxiety_score": scores.get("Anxiety_Score"),
                "depression_score": scores.get("Depression_Score"),
                "survey_responses": survey_responses_dict  # Store all answers as JSON
            }
            
            # Insert data into your Supabase table (e.g., 'user_info')
            supabase.table('user_info').insert(db_data).execute()
            
            print(f"Data successfully saved to Supabase for {name}")
            st.toast("Results securely saved.", icon="🔒") # Friendly notification

        except Exception as e:
            print(f"Supabase save error: {e}")
            st.warning(f"Could not save results to secure database: {e}")
    else:
        # This will show if the st.secrets are missing or connection failed
        st.warning("Database connection not found. Results not saved securely.")
    # --- END ADDED ---


# --- Initialize Session State ---
# This dictionary holds the current state of the app
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1 # Start at step 1
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'image_buffer' not in st.session_state:
    st.session_state.image_buffer = None
if 'survey_responses' not in st.session_state:
    st.session_state.survey_responses = {} # Store survey answers here
if 'eeg_simulated' not in st.session_state:
    st.session_state.eeg_simulated = False # Track if EEG box was checked
if 'results' not in st.session_state:
    st.session_state.results = None # Store final results to avoid recalculating


# --- MAIN PAGE UI ---
st.set_page_config(layout="wide")
st.title("🧠 Multimodal Depression Screening Tool")
st.markdown("Follow the steps below. **This tool is not a medical diagnosis.**")
st.markdown("---")

# --- Step Navigation Logic ---
def go_to_step(step_number):
    st.session_state.current_step = step_number
    # Clear results when going back from the results page
    if step_number < 5 and st.session_state.results is not None:
         st.session_state.results = None

# --- Display Content Based on Current Step ---

# STEP 1: Consent & Info
if st.session_state.current_step == 1:
    st.subheader("Step 1: Consent & Name")
    consent = st.checkbox("I consent and confirm I am over 18. I understand my data will be saved.", key="consent_check_step1")
    st.session_state.user_name = st.text_input("Please enter your Name (or a unique ID)*", key="name_input_step1", value=st.session_state.user_name)

    if st.button("Next: Capture Image", type="primary", key="next_step1"):
        if consent and st.session_state.user_name:
            go_to_step(2)
            st.rerun()
        elif not consent: st.error("Please provide consent.")
        else: st.error("Please enter your name or ID.")

# STEP 2: Capture Image
elif st.session_state.current_step == 2:
    st.subheader("Step 2: Capture Snapshot")
    st.write(f"Hello {st.session_state.user_name}, please look directly at the camera.")
    img_buf = st.camera_input("Take a picture 📸", key="camera_step2")

    if img_buf is not None:
        st.session_state.image_buffer = img_buf
        st.success("✅ Snapshot captured.")
    elif st.session_state.image_buffer is not None:
        st.success("✅ Snapshot already captured.")

    col1, col2 = st.columns([1, 1])
    with col1: st.button("Back: Consent & Info", on_click=go_to_step, args=(1,), key="back_step2")
    with col2: st.button("Next: Survey", on_click=go_to_step, args=(3,), type="primary", disabled=(st.session_state.image_buffer is None), key="next_step2")

# STEP 3: Survey
elif st.session_state.current_step == 3:
    st.subheader("Step 3: Questionnaire (DASS-21)")
    st.write("Select the option indicating how much each statement applied to you **over the past week**.")

    current_responses = st.session_state.survey_responses
    all_answered = True

    with st.expander("Show DASS-21 Questions", expanded=True):
        col1, col2 = st.columns(2)
        for i, q_text in enumerate(questions_text):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                default_index = current_responses.get(q_text, None)
                selected_opt = st.radio(f"{i+1}. {q_text}", options=response_options, key=f"q_{i}_step3", index=default_index, horizontal=False)
                current_responses[q_text] = int(selected_opt.split(":")[0]) if selected_opt else None
                if current_responses[q_text] is None: all_answered = False

    st.session_state.survey_responses = current_responses

    if all_answered: st.success("✅ All questions answered.")
    else: st.warning("Please answer all 21 questions to proceed.")

    col1, col2 = st.columns([1, 1])
    with col1: st.button("Back: Capture Image", on_click=go_to_step, args=(2,), key="back_step3")
    with col2: st.button("Next: EEG (Optional)", on_click=go_to_step, args=(4,), type="primary", disabled=(not all_answered), key="next_step3")

# STEP 4: EEG (Optional Simulation)
elif st.session_state.current_step == 4:
    st.subheader("Step 4: EEG Connection (Optional Simulation)")
    st.write("This step simulates connecting an EEG device. If you had the device, this is where signals would be captured.")
    
    # --- *** MODIFIED EEG CHECK *** ---
    simulate_eeg_check = False
    if eeg_model: # Only show checkbox if the EEG model loaded
        simulate_eeg_check = st.checkbox("Simulate EEG Device Connection (Check this box to include simulated EEG results)", 
                                         key="eeg_check_step4", 
                                         value=st.session_state.eeg_simulated)
        if simulate_eeg_check:
            st.info("✅ EEG Device Connection Simulated. Results will be generated based on other inputs.")
        else:
            # Add a message explaining no *real* check is happening
            st.info("ℹ️ EEG Simulation is OFF. The app cannot detect physical devices. Check the box only for demo purposes.")
    else:
        st.warning("`eeg_model.pkl` not found. EEG simulation unavailable.")
        simulate_eeg_check = False # Force false if no model

    # Store the choice
    st.session_state.eeg_simulated = simulate_eeg_check
    # --------------------------------

    col1, col2 = st.columns([1, 1])
    with col1: st.button("Back: Survey", on_click=go_to_step, args=(3,), key="back_step4")
    with col2:
        if st.button("Analyze & View Results", type="primary", key="analyze_step4"):
            st.session_state.results = None # Clear previous results
            go_to_step(5)
            st.rerun()

# STEP 5: Results
elif st.session_state.current_step == 5:
    st.subheader("Step 5: Analysis & Results")

    # Perform analysis only once
    if not st.session_state.results:
        # Retrieve needed data from session state
        name = st.session_state.user_name
        image_buffer = st.session_state.image_buffer
        survey_responses = st.session_state.survey_responses
        simulate_eeg = st.session_state.eeg_simulated
        
        if not name or not image_buffer or not survey_responses:
            st.error("Error: Missing data from previous steps. Please restart.")
            if st.button("Start Over"):
                keys_to_clear = list(st.session_state.keys())
                for key in keys_to_clear: del st.session_state[key]
                st.rerun()
        else:
            with st.spinner("Analyzing your results..."):
                pil_image = Image.open(image_buffer)
                emotion = analyze_image(pil_image)

                responses_coded = {dass_mapping[q]: v for q, v in survey_responses.items()}
                scores = calculate_scores(responses_coded)
                
                input_data_dict = {
                    **responses_coded,
                    "Stress_Score": scores["Stress_Score"],
                    "Anxiety_Score": scores["Anxiety_Score"]
                }
                
                input_df = pd.DataFrame([input_data_dict])
                try: 
                    input_df_ordered = input_df[feature_cols_survey_model]
                    pred_val = survey_model.predict(input_df_ordered)[0]
                    survey_label = "Likely Depressed" if pred_val == 1 else "Not Depressed"
                except Exception as e:
                    st.error(f"Survey prediction error: {e}"); survey_label = "Error"
                
                eeg_prediction = "N/A"
                if simulate_eeg and eeg_model:
                    # SIMULATE: Use survey result for demo prediction
                    eeg_prediction = "Depressed (Simulated)" if survey_label == "Likely Depressed" else "Normal (Simulated)"
                    # In a real app: eeg_prediction = eeg_model.predict(processed_eeg_data)
                elif simulate_eeg and not eeg_model:
                    eeg_prediction = "Error: EEG Model not loaded"
                
                # Save results (CSV and image)
                # This function now saves to BOTH CSV and Supabase
                save_results(name, pil_image, responses_coded, scores, survey_label, emotion, eeg_prediction)
                
                # Store results in session state
                st.session_state.results = {
                    "scores": scores, "survey_label": survey_label,
                    "emotion": emotion, "eeg_prediction": eeg_prediction
                }
                
                st.success("✅ Analysis Complete!")
                # Rerun immediately to display the results
                st.rerun() 

    # --- Display results if they exist in session state ---
    if st.session_state.results:
        scores = st.session_state.results["scores"]
        survey_label = st.session_state.results["survey_label"]
        emotion = st.session_state.results["emotion"]
        eeg_prediction = st.session_state.results["eeg_prediction"]

        st.subheader("Your Screening Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Stress Score", scores['Stress_Score'])
            st.metric("Anxiety Score", scores['Anxiety_Score'])
            st.metric("Depression Score", scores['Depression_Score'])
        with res_col2:
            st.metric("Survey Prediction", survey_label)
            st.metric("Detected Emotion", emotion)
        with res_col3:
            st.metric("EEG Prediction", eeg_prediction)
        
        st.markdown("---")
        st.subheader("Suggestions & Next Steps")
        get_suggestions(survey_label, emotion, scores, eeg_prediction)

        # Button to start over
        if st.button("Start New Screening"):
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear: del st.session_state[key]
            st.rerun()

# --- TAB 2: VIEW RESULTS (Unchanged) ---
# ... (Keep the View My Results tab code exactly as it was) ...