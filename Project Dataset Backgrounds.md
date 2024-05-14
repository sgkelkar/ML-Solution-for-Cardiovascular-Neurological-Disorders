## Problem Set 1 Data Description

The “Heart Disease Databases” are a collection of 4 databases gathered across five medical centers in the 1980s. Each database contains clinical information and results of diagnostic exams regarding patients who were admitted to the hospital with severe cardiovascular impairments. Every database includes one entry per admitted patient. The databases were gathered and created by the following investigators:

- Hungarian Institute of Cardiology, Budapest, Hungary: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach, CA and Cleveland Clinic Foundation, Cleveland, OH: Robert Detrano, M.D., Ph.D.

The Heart Disease Databases were donated by David W. Aha (aha@ics.uci.edu) in 1988 and can be retrieved at the URL: [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

The Heart Disease Databases are provided in this assignment packet as four *.csv files, roughly one per medical center involved in the study (i.e., cleveland.csv, switzerland.csv, va.csv, and hungarian.csv). Each file includes a subset of 14 features from the original database, which are detailed below:

- age: Age of the patient (in years).
- sex: Gender of the patient (1=male; 0=female).
- cp: Type of chest pain reported by the patient (1=typical angina; 2=atypical angina; 3=nonanginal pain; 4=asymptomatic).
- trestbps: Resting blood pressure (in mm Hg) measured at the time of admission to the hospital.
- chol: Serum cholesterol (in mg/dl).
- fbs: Fasting blood sugar above 120 mg/dl (1=true; 0=false).
- restecg: Resting electrocardiographic results (0=normal; 1=having an ST-T wave abnormality, i.e., T-wave inversions and/or ST elevation or depression greater than 0.05 mV; 2=showing probable or definite left ventricular hypertrophy according to the Estes’ criteria).
- thalach: Maximum heart rate (in bpm) achieved.
- exang: Angina induced by exercise (1=yes; 0=no).
- oldpeak: ST depression induced by exercise relative to rest.
- slope: Slope of the peak exercise ST segment (1=up-sloping; 2=flat; 3=down-sloping).
- ca: Number of major vessels colored by fluoroscopy. Possible values: 0, 1, 2, 3.
- thal: History of thalassemia (3=normal; 6=fixed defect; 7=reversible defect).
- num: Angiographic disease status (0= normal status, i.e., <50% diameter narrowing; 1, 2, 3, 4= disease status, i.e., >50% diameter narrowing).

A description of the databases can be found in [1] and is included in this assignment packet as a PDF file (file: Detrano 1989.pdf). A total of 920 entries are included across the four databases, with the feature “num” reporting the results of the coronary angiogram performed on each patient. If the coronary angiogram showed 50% or more luminal narrowing in one or more major epicardial vessel, the patient was diagnosed with coronary artery disease, and num would report the disease’s severity (scale: 1 through 4, where 4 indicates the highest severity). Vice versa, num was set to 0 for those patients whose coronary angiogram indicated less than 50% luminal narrowing in all major epicardial vessels. Several features have missing values, which are indicated by symbols “?”, “-9”, or “NaN”.



## Problem Set 2 (Regression)

### Description of the Data

The “Parkinson’s Data Set” comprises voice measurements from patients with early-stage Parkinson’s disease (PD) who participated in a six-month trial evaluating a tele-monitoring device for remote symptom progression monitoring. These recordings were automatically captured in patients’ homes.

This dataset was developed by Dr. Athanasios Tsanas and Dr. Max Little of the University of Oxford, UK, in collaboration with 10 medical centers in the USA and Intel Corporation, which developed the tele-monitoring device for recording speech signals. The Parkinson’s Data Set was donated by the authors in 2008 and can be accessed at the URL: [Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons).

The Parkinson’s Data Set is provided as one *.csv file (dataset parkinson.csv) and includes 19 features, detailed as follows:

- name: Integer uniquely identifying each subject participating in the study.
- Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP: Five measures of variation in fundamental frequency (in Hz) of the speech sound.
- Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA: Six measures of variation in amplitude of the speech sound.
- NHR, HNR: Two measures of the ratio between noise and tonal components in the voice.
- RPDE: Measure of nonlinear dynamical complexity of the speech sound.
- DFA: Signal fractal scaling exponent of the speech sound.
- PPE: Nonlinear measure of the variation in fundamental frequency of the speech sound.
- motor UPDRS: UPDRS motor score, a numerical score assigned by the clinician upon examining the motor abilities of the patient in a sequence of predefined motor tasks.
- total UPDRS: UPDRS total score, the sum of multiple scores assigned by the clinician upon examining the motor abilities, mental abilities, and mood of the patient in a sequence of predefined tasks.

A description of the dataset can be found in [2] and is included in this assignment packet as a PDF file (file: Tsanas 2010.pdf). The dataset comprises 5,875 voice recordings from 42 patients, with approximately 200 recordings per patient. No missing values are reported.
