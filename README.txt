# Multimodal Depression Screening Tool - Execution Guide

This guide explains how to set up and run the Streamlit application for depression screening.

## 1. Required Files

Make sure you have the following files in the **same project folder**:

1.  `app.py`          (The Streamlit application code)
2.  `requirements.txt` (List of Python libraries needed)
3.  `random_forest_model_corrected.joblib` (The trained DASS-21 survey model)
4.  `facial_emotion_model.h5`     (The trained facial emotion image model)
5.  `eeg_model.pkl`       (OPTIONAL: The trained EEG model for simulation)

## 2. Setup Instructions

You need Python 3 installed on your system.

   a. **Open Terminal/Command Prompt:** Open your command line interface (like Command Prompt, PowerShell, or Terminal).

   b. **Navigate to Project Folder:** Use the `cd` command to go into the folder where you saved the files listed above.
      Example:
      ```bash
      cd C:\Users\Naveena harikrishnan\Downloads\our_mini_project
      ```

   c. **Install Dependencies:** Run the following command exactly once to install all the necessary Python libraries listed in the `requirements.txt` file. You need an internet connection for this.
      ```bash
      pip install -r requirements.txt
      ```
      Wait for the installation to complete. If you encounter errors, check the terminal output for details (e.g., issues with Python version or specific libraries).

## 3. Running the Application

   a. **Ensure you are in the project folder** in your terminal/command prompt.

   b. **Run the Streamlit command:**
      ```bash
      streamlit run app.py
      ```

   c. **Open in Browser:** Streamlit will automatically open the application in your default web browser. If it doesn't, the terminal will provide a "Local URL" (usually `http://localhost:8501`) that you can copy and paste into your browser.

## 4. Using the Application

   a. **Follow the Steps:** The application guides you through the steps: Consent & Name -> Capture Image -> Survey -> EEG (Optional) -> Results.
   b. **Models:** The app will use the loaded survey and image models. The EEG model step is optional and simulated (check the box to include its simulated output).
   c. **Saving Data:** Results (including the image snapshot filename) are saved locally to `survey_responses_streamlit.csv`. Image snapshots are saved in the `captured_images_streamlit` folder.
   d. **Viewing Results:** You can view past results using the "View My Results" tab by entering the name used during submission.

## Notes

* Ensure your webcam is working and accessible to your browser for the image capture step.
* The EEG model (`eeg_model.pkl`) is optional. If the file is not found, the app will still run but will disable the EEG simulation step.