# MVTec AD Anomaly Detector - Streamlit Prototype (Group 3)

A Streamlit web application that turns the Random Forest model into an interactive tool.
It has five pages: **Home**, **Data Dashboard**, **Model Performance**, **Make Prediction** and **About**.

## Folder structure

```
mvtec_streamlit_app/
|-- app.py                       # the Streamlit application
|-- requirements.txt             # Python packages
|-- README.md                    # this file
|-- models/
|   |-- best_model_random_forest.joblib   # saved model      
|   |-- feature_columns.joblib            # training columns 
|   `-- feature_scaler.joblib             # scaler           
|-- data/
|   |-- cleaned_project_dataset.csv       # cleaned metadata (3,260 images)
|   `-- predictions.db                    # created automatically (prediction history)
`-- screenshots/                          # application screenshots here
```

## 1. Model files

The three `.joblib` files are already in the `models/` folder. They were saved with
scikit-learn 1.6.1, so `requirements.txt` pins that version - keep it to avoid version-mismatch warnings.

## 2. Run locally

``n app.py
````bash
python -m venv venv
venv\Scripts\activate          # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py

The app opens at http://localhost:8501.

## 3. Using the app

* **Make Prediction -> Upload an image:** choose the object category, upload a PNG/JPG, press **Predict**.
  The app computes the same nine visual features used in training (128x128 resize; mean/std of R, G, B;
  brightness; contrast; edge density) and shows the probability that the component is anomalous.
* **Make Prediction -> Enter feature values manually:** type the feature values (each field has a valid range) and press **Predict**.
* Every prediction is saved (numbers only, no images) in `data/predictions.db` and shown in the history table.

## 4. Live link(Deployed online)
1.  Github repo : https://github.com/Hades3942/Industrial-Anomaly-detection-app
## 5. Notes
* Dataset licence: MVTec AD is CC BY-NC-SA 4.0 (non-commercial use with attribution).
