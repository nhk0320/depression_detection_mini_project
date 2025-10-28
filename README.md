# Multimodal Depression Screening Tool

This Streamlit application provides a preliminary screening for depression symptoms using the DASS-21 survey, facial emotion detection from a webcam snapshot, and an optional simulated EEG input.

**Disclaimer:** This tool is for demonstration and informational purposes only and is **not** a substitute for a professional medical diagnosis. If you have concerns about your mental health, please consult a qualified healthcare provider.

---
## 1. Required Files

Ensure the following files are present in the same project folder:

* `app.py`: The main Streamlit application script.
* `requirements.txt`: Lists the required Python libraries.
* `random_forest_model_corrected.joblib`: The trained DASS-21 survey prediction model.
* `facial_emotion_model.h5`: The trained facial emotion image classification model.
* `eeg_model.pkl`: (Optional) The trained EEG prediction model (used for simulation).

---
## 2. Setup Instructions

You need Python 3 installed on your system (preferably Python 3.9, 3.10, or 3.11 for best compatibility with libraries like TensorFlow).

   a. **Open Terminal/Command Prompt:** Navigate to your command line interface.

   b. **Navigate to Project Folder:** Use the `cd` command to change the directory to where you saved the project files.
      ```bash
      cd path/to/your/project_folder
      ```
      *(Replace `path/to/your/project_folder` with the actual path, e.g., `C:\Users\Naveena harikrishnan\Downloads\our_mini_project`)*

   c. **Install Dependencies:** Run the following command once to install all necessary libraries. An internet connection is required.
      ```bash
      pip install -r requirements.txt
      ```
      Wait for the installation to finish.

---
## 3. Running the Application

   a. **Ensure you are in the project folder** in your terminal.

   b. **Execute the Streamlit command:**
      ```bash
      streamlit run app.py
      ```

   c. **Open in Browser:** Streamlit will automatically open the application in your web browser, usually at `http://localhost:8501`.

---
## 4. How to Use the App

   a. **Follow Steps:** The application is designed as a step-by-step process:
      1.  **Consent & Name:** Provide consent and enter your name/ID.
      2.  **Capture Image:** Use your webcam to take a snapshot.
      3.  **Survey:** Answer all 21 DASS-21 questions.
      4.  **EEG (Optional):** Check the box if you want to *simulate* using the EEG model for the final result.
      5.  **Results:** View the analysis from the survey, image, and (if simulated) EEG models, along with suggestions.

   b. **Data Saving:** Each submission saves the responses, predictions, and the image filename to `survey_responses_streamlit.csv`. The actual snapshot image is saved locally in the `captured_images_streamlit` folder. *(Note: On deployed versions like Streamlit Community Cloud, these local files are temporary).*

   c. **View Past Results:** Use the "View My Results" tab to look up previous submissions by the name/ID used.

---
## 5. Important Notes

* **Webcam:** Ensure your webcam is connected and your browser has permission to access it for the snapshot step.
* **EEG Model:** The `eeg_model.pkl` file is optional. If not present, the app will still function, but the EEG simulation step will be disabled. The app **cannot** detect or read data from a real EEG device connected via USB.
