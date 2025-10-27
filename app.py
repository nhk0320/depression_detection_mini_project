import streamlit as st
import pandas as pd
import joblib
import os
# No longer needed if not scaling: from sklearn.preprocessing import StandardScaler

# --- Configuration ---
MODEL_PATH = "random_forest_model_corrected.joblib" # Use the corrected model
# SCALER_PATH = "scaler.joblib" # SCALER NOT NEEDED if model trained without scaling
RESPONSES_CSV = "survey_responses.csv"

# --- Load Model (No Scaler) ---
@st.cache_resource # Cache resource loading
def load_model(): # Renamed function for clarity
    try:
        model = joblib.load(MODEL_PATH)
        # scaler = None # No scaler to load
        return model # Return only the model
    except FileNotFoundError:
        st.error(f"Error: Model ('{MODEL_PATH}') file not found.")
        st.error("Please ensure you have saved the CORRECTED model using joblib.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

# --- DASS-21 Questions and Mapping (Copied from your notebook cell #12) ---
# --- (Keep the full dass_mapping and response_options as before) ---
dass_mapping = {
    # Stress questions
    "i found it hard to wind down": "Q3_1_S1",
    "I tended to over-react to situations.": "Q3_2_S2",
    "I felt that I was using a lot of nervous energy.": "Q3_3_S3",
    "I found myself getting agitated.": "Q3_4_S4",
    "I found it difficult to relax.": "Q3_5_S5",
    "I was intolerant of anything that kept me from getting on with what I was doing.": "Q3_6_S6",
    "I felt that I was rather touchy.": "Q3_7_S7",

    # Anxiety questions
    "I was aware of dryness of my mouth.": "Q3_8_A1",
    "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness without physical exertion).": "Q3_9_A2",
    "I experienced trembling (e.g., in the hands).": "Q3_10_A3",
    "I was worried about situations in which I might panic and make a fool of myself.": "Q3_11_A4",
    "I felt I was close to panic.": "Q3_12_A5",
    "I was aware of the action of my heart without physical exertion (e.g., heart rate increase).": "Q3_13_A6",
    "I felt scared without any good reason.": "Q3_14_A7",

    # Depression questions
    "I couldn’t seem to experience any positive feeling at all": "Q3_15_D1",
    "I found it difficult to work up the initiative to do things.": "Q3_16_D2",
    "I felt that I had nothing to look forward to.": "Q3_17_D3",
    "I felt down-hearted and blue.": "Q3_18_D4",
    "I was unable to become enthusiastic about anything.": "Q3_19_D5",
    "I felt I wasn’t worth much as a person.": "Q3_20_D6",
    "I felt that life was meaningless.": "Q3_21_D7"
}
dass_mapping = {k.strip(): v for k, v in dass_mapping.items()}
questions_text = list(dass_mapping.keys())
response_options = [
    "0: Did not apply to me at all",
    "1: Applied to me to some degree, or some of the time",
    "2: Applied to me to a considerable degree, or a good part of time",
    "3: Applied to me very much, or most of the time"
]


# --- Feature columns the CORRECTED model expects ---
# (21 Questions + Stress_Score + Anxiety_Score)
feature_cols_model = [
    'Q3_1_S1', 'Q3_2_S2', 'Q3_3_S3', 'Q3_4_S4', 'Q3_5_S5', 'Q3_6_S6', 'Q3_7_S7',
    'Q3_8_A1', 'Q3_9_A2', 'Q3_10_A3', 'Q3_11_A4', 'Q3_12_A5', 'Q3_13_A6', 'Q3_14_A7',
    'Q3_15_D1', 'Q3_16_D2', 'Q3_17_D3', 'Q3_18_D4', 'Q3_19_D5', 'Q3_20_D6', 'Q3_21_D7',
    'Stress_Score', 'Anxiety_Score' # NO Depression_Score
]

# --- Helper Functions (Keep calculate_scores and save_data as before) ---
def calculate_scores(responses_coded):
    """Calculates Stress, Anxiety, and Depression scores from coded responses."""
    stress_cols = [f"Q3_{i}_S{j}" for i, j in zip(range(1, 8), range(1, 8))]
    anxiety_cols = [f"Q3_{i}_A{j}" for i, j in zip(range(8, 15), range(1, 8))]
    depression_cols = [f"Q3_{i}_D{j}" for i, j in zip(range(15, 22), range(1, 8))]

    scores = {
        "Stress_Score": sum(responses_coded.get(col, 0) for col in stress_cols),
        "Anxiety_Score": sum(responses_coded.get(col, 0) for col in anxiety_cols),
        "Depression_Score": sum(responses_coded.get(col, 0) for col in depression_cols)
    }
    return scores

def save_data(user_info, responses_coded, scores, prediction):
    """Saves the collected data to a CSV file."""
    data_to_save = {**user_info, **responses_coded, **scores, "Prediction": prediction}
    df_new_row = pd.DataFrame([data_to_save])

    if not os.path.exists(RESPONSES_CSV):
        df_new_row.to_csv(RESPONSES_CSV, index=False)
    else:
        df_new_row.to_csv(RESPONSES_CSV, mode='a', header=False, index=False)

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🧠 DASS-21 Depression Screening Tool")
st.markdown("---")

# --- Step 1: Consent/Permission ---
st.header("1. Consent")
consent = st.checkbox(
    "I understand this is a screening tool and not a diagnosis. "
    "I consent to providing my responses. (Conceptually, this includes camera permission if needed later)"
)
st.markdown("---")

if consent:
    # Load model only after consent
    model = load_model() # Only load the model

    if model: # Proceed only if model loaded successfully
        # --- Step 2: Questionnaire Form ---
        st.header("2. Questionnaire")
        st.write("Please read each statement and select the option (0, 1, 2, or 3) which indicates how much the statement applied to you **over the past week**.")

        with st.form("dass21_form"):
            st.subheader("Your Information")
            name = st.text_input("Enter your Name*", key="name_input")
            # user_id = st.text_input("Enter a unique ID (Optional)", key="id_input")

            st.subheader("Questions")
            responses = {}
            cols = st.columns(2) # Display questions in 2 columns

            for i, q_text in enumerate(questions_text):
                with cols[i % 2]: # Alternate columns
                    selected_option = st.radio(
                        f"{i+1}. {q_text}",
                        options=response_options,
                        key=f"q_{i}",
                        horizontal=False
                    )
                    responses[q_text] = int(selected_option.split(":")[0])

            submitted = st.form_submit_button("Submit & See Results")

            if submitted:
                # --- Step 3: Validation and Processing ---
                if not name:
                    st.error("⚠️ Please enter your name before submitting.")
                else:
                    st.success("Responses Submitted! Calculating results...")

                    responses_coded = {dass_mapping[q]: responses[q] for q in responses}
                    scores = calculate_scores(responses_coded)

                    # Prepare data for the model (including calculated Stress & Anxiety scores)
                    input_data_dict = {**responses_coded,
                                       "Stress_Score": scores["Stress_Score"],
                                       "Anxiety_Score": scores["Anxiety_Score"]}
                    input_df = pd.DataFrame([input_data_dict])

                    # Ensure columns are in the EXACT order the model expects
                    try:
                        # Use the corrected feature list (without Depression_Score)
                        input_df_ordered = input_df[feature_cols_model]
                    except KeyError as e:
                        st.error(f"Data Preparation Error: Missing expected feature column: {e}")
                        st.error("Please check the 'feature_cols_model' list and the 'dass_mapping'.")
                        st.stop()

                    # --- SCALING REMOVED ---
                    # No scaling needed if the model was trained on unscaled data
                    # try:
                    #     input_scaled = scaler.transform(input_df_ordered)
                    # except Exception as e:
                    #     st.error(f"Data Scaling Error: {e}")
                    #     st.stop()

                    # Predict using the loaded model with UNSCALED data
                    try:
                        # Use input_df_ordered directly (unscaled)
                        prediction_val = model.predict(input_df_ordered)[0]
                        prediction_label = "Likely Depressed" if prediction_val == 1 else "Not Depressed"
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                        st.stop()

                    # --- Step 4: Display Results (Keep as is) ---
                    st.header("3. Your Screening Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Stress Score", scores['Stress_Score'])
                        st.metric("Anxiety Score", scores['Anxiety_Score'])
                        st.metric("Depression Score", scores['Depression_Score']) # Still display this score

                    with col2:
                        st.subheader("Prediction:")
                        if prediction_label == "Likely Depressed":
                            st.warning(f"**{prediction_label}**")
                        else:
                            st.success(f"**{prediction_label}**")

                    st.info(
                        "**Disclaimer:** This tool provides a preliminary screening... (keep disclaimer)"
                    )
                    st.markdown("---")


                    # --- Step 4.5: Add Suggestions if Likely Depressed (Keep as is) ---
                    if prediction_label == "Likely Depressed":
                        st.subheader("Suggestions & Next Steps")
                        st.warning("**Important:** This screening suggests you might be experiencing symptoms consistent with depression. Please remember this is not a diagnosis, but it's worth exploring these feelings further. Consider the following steps:")
                        st.markdown("""
                            * **Seek Professional Help:** This is the most effective step you can take. Schedule an appointment with your doctor 👨‍⚕️, a therapist 🧠, counselor, or a mental health professional. They are trained to understand what you're going through, can provide an accurate diagnosis if needed, and discuss the best treatment options for you, which might include therapy (like CBT or talk therapy), medication, or lifestyle adjustments. Don't hesitate – reaching out is a sign of strength. 🩺

                            * **Talk to Someone You Trust:** Sharing your feelings can make a significant difference. Reach out to a close friend 🧑‍🤝‍🧑, family member 👨‍👩‍👧‍👦, partner ❤️, or anyone you feel comfortable confiding in. Letting someone know how you feel can lessen the burden and provide emotional support. You don't have to go through this alone. 🤗

                            * **Focus on Self-Care Basics:** When feeling down, even basic routines can feel challenging, but they are important foundations for well-being.
                                * **Sleep:** Aim for a consistent sleep schedule (7-9 hours is often recommended). Create a relaxing bedtime routine if possible. 😴🌙
                                * **Nutrition:** Try to eat regular, balanced meals. Good nutrition fuels your body and mind. Even small, healthy choices can help. 🍎🥦
                                * **Movement:** Physical activity releases endorphins, which can naturally boost your mood. Start small – a short daily walk outdoors can be very beneficial. 🚶‍♀️☀️

                            * **Practice Mindfulness & Relaxation:** Stress and anxiety often accompany depression. Simple techniques practiced regularly can help calm your mind. Try deep breathing exercises 😮‍💨, guided meditations (many apps and online resources are available), or gentle yoga 🧘.

                            * **Engage (Gently) in Activities:** Anhedonia (loss of interest in enjoyable activities) is common in depression. While motivation might be low, try to gently re-engage with hobbies or simple pleasures you used to enjoy. This could be listening to music 🎶, spending time in nature 🌳, reading 📚, or a creative pursuit 🎨. Start with low-pressure activities for short periods.

                            * **Limit Alcohol & Substances:** While they might seem like a temporary escape, alcohol and recreational drugs often worsen depression symptoms in the long run and can interfere with treatment. Consider reducing or avoiding them. 🚫🍺

                            * **Be Patient and Kind to Yourself:** Healing and recovery take time; there will be good days and difficult days. Avoid self-criticism and acknowledge that feeling this way is not your fault. Celebrate small victories and be patient with your progress. ❤️⏳

                            **Remember, you are not alone, many people experience these feelings, and effective help is available.** Taking the first step to reach out is often the hardest but most important one. 💪🫂
                        """)
                        st.error("**Crucially, please consult a healthcare professional or a mental health expert for a proper evaluation, diagnosis, and personalized guidance.**")
                        st.markdown("---")

                    # --- Step 5: Save Data (Keep as is) ---
                    user_info = {"Name": name}
                    try:
                        save_data(user_info, responses_coded, scores, prediction_label)
                        st.success(f"Response for {name} recorded successfully in '{RESPONSES_CSV}'.")
                    except Exception as e:
                        st.error(f"Could not save data to CSV: {e}")

    else:
        st.warning("Cannot proceed without loading the prediction model.") # Updated message

else:
    st.info("Please check the consent box above to begin the questionnaire.")

st.markdown("---")
st.caption("DASS-21 Screening Tool v1.0")