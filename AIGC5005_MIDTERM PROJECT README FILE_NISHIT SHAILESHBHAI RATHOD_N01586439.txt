--------------------------------------------------
NAME: NISHIT SHAILESHBHAI RATHOD
STUDENT NUMBER: N01586439
COURSE: AIGC 5005 AI CAPSTONE PROJECT PREPARATION
--------------------------------------------------
README FILE - MIDTERM PROJECT
--------------------------------------------------
STEPS TO TAKE IN ORDER TO RUN THE PYTHON PROGRAMS
--------------------------------------------------


STEP 1: OPEN JUPYTER NOTEBOOK FROM ANACONDA(3) NAVIGATOR

STEP 2: MAKE A FOLDER

STEP 3: UPLOAD THE "AIGC5005_MIDTERM PROJECT PYTHON PROGRAM FILE_ NISHIT SHAILESBHAI RATHOD_N01586439.ipynb" NAMED FILE AND THE risk-train.csv & risk-test.csv NAMED FILE INTO THE FOLDER.

STEP 4: OPEN THE UPLOADED .IPYNB FILE

STEP 4: DOWNLOAD THE REQUIRED PYTHON LIBRARIES USING ANACONDA PROMPT
----------------------
PYTHON LIBRARIES USED
----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

-------------------------
STEP 5: READ THE risk-train.csv FILE
- CODE: df = pd.read_csv("risk-train.csv")

-------------------------
STEP 6: ONCE YOU READ THE CSV FILE YOU ARE READY TO RUN CODES FOR THE FOLLOWING TOPICS - 
- UNDERSTANDING AND KNOWING THE DATA
- CLEANING AND PREPROCESSING THE DATA
- KNOWING THE DESCRIPTIVE STATISTICAL SUMMARY
- EXPLORATORY DATA ANALYSIS
- FEATURE SELECTION BASED ON SELECTOR SCORE AND P-VALUE
- SCALING THE DATA
- SPLITING THE DATA FOR TRAINING AND TESTING
- IMPLEMENTING VARIOUS CLASSIFICATION MODELS
- EVALUATING THE MODELS .VIA CONFUSION MATRIX, CLASSIFICATION REPORT, ROC CURVE AND AUC

-------------------------
STEP 7: READ THE risk-test.csv FILE.

STEP 8: TRANSFORM THE TEST DATA EXACTLY THE SAME WAY THAT WE DID FOR THE TRAIN DATA JUST BY IMPLEMENTING SAME DATA CLEANING AND PREPROCESSING TECHNIQUES WHICH WERE CODED FOR risk-train.csv INITIALLY.

STEP 9: SELECT THE BEST CLASSIFICATION MODEL FROM THE MODELS THAT WE IMPLEMENTED EARLIER  
 
STEP 10: USE THAT MODEL ON risk-test.csv AND PREDICT THE CLASS.

STEP 11: AFTER PREDICTING THE CLASSES, SAVE THE PREDICTED_CLASS INTO A DATAFRAME ALONG WITH THE ORDER_ID AND CONVERT THE DATAFRAME TO .txt FILE.
