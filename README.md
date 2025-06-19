W.I.P Project
# Machine Learning / Deep Learning Intrusion Detection System (IDS

Collaboration effort by Quy Nguyen, Ngan Kha and Yen Do

# Project Description:
- This project implements a Machine Learning based Intrusion Detection System using the CIC-IDS2017 dataset. Currently, there is only the Random Forest Model instead of having 2 more, consisting of KNN and SVM models, which were taken out for a remaster.

# Structure:
- data folder: contains two more folders, one containing the original dataset downloaded from the Canadian Institute for Cybersecurity and a cleaned_dataset folder, created during the preprocess procedure.
- script folder: contains all scripts of the project, including:
+ preprocess.py: perform data cleaning (ensuring NaN, rounding to a definite value, and dropping duplicates)
+ machine learning models: Python scripts that uses specific ML models such as Random Forest. Other scripts are not yet included.
+ combined_evaluation_results.py: script used to display all evaluation metrics my all the models for easier comparision, though only random forest is read as of right now.

# Setup:
- Please be in the main directory of the project in your terminal, and setup a virtual environment such as venv.
- From there, ensure that specific packages are installed, including these (subject to change as project grows):
+ pandas
+ numpy
+ scikit-learn
+ matplotlib
+ seaborn
+ jupyter
+ tabulate

- Ensure that the .csv dataset are extracted in the data folder so that preprocess.py can read them one by one, so that the data directory should display like this:
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX
Friday-WorkingHours-Morning.pcap_ISCX
Monday-WorkingHours.pcap_ISCX
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX
Tuesday-WorkingHours.pcap_ISCX
Wednesday-workingHours.pcap_ISCX
(and any additional content that you may have, such as backup folders)

- Navigate to the script folder, and simply run python3 preprocess.py first. Once done, you can run python3 random_forest.py or any existing models in no particular order, as long as they are ran after preprocess.py and before combined_evaluation_results.py. 
- Running python3 combined_evaluation_results.py will display your model results, which a sample of a random forest run can be seen below:
+----+----------------------------+----------+--------+-----------+----------+
|    |        Attack Type         | Accuracy | Recall | Precision | F1-Score |
+----+----------------------------+----------+--------+-----------+----------+
| 0  |           BENIGN           |  0.9999  | 0.7666 |  0.7667   |  0.7667  |
| 1  |            Bot             |  0.9705  | 0.4852 |    0.5    |  0.4925  |
| 2  |            DDoS            |  0.9998  | 0.4999 |    0.5    |   0.5    |
| 3  |       DoS-GoldenEye        |  0.9985  | 0.3328 |  0.3333   |  0.3331  |
| 4  |          DoS-Hulk          |  0.9998  | 0.3333 |  0.3333   |  0.3333  |
| 5  |      DoS-Slowhttptest      |  0.9952  | 0.3317 |  0.3333   |  0.3325  |
| 6  |       DoS-slowloris        |  0.9907  | 0.3302 |  0.3333   |  0.3318  |
| 7  |        FTP-Patator         |   1.0    |  1.0   |    1.0    |   1.0    |
| 8  |         Heartbleed         |   1.0    |  1.0   |    1.0    |   1.0    |
| 9  |        Infiltration        |  0.7143  | 0.3571 |    0.5    |  0.4167  |
| 10 |          PortScan          |  0.9998  | 0.4999 |    0.5    |   0.5    |
| 11 |        SSH-Patator         |  0.9891  | 0.4946 |    0.5    |  0.4973  |
| 12 |  Web-Attack- -Brute-Force  |  0.6446  | 0.1888 |  0.2916   |  0.2292  |
| 13 | Web-Attack- -Sql-Injection |  0.625   | 0.2708 |  0.4166   |  0.3254  |
| 14 |      Web-Attack- -XSS      |  0.5916  | 0.1972 |  0.3333   |  0.2476  |
+----+----------------------------+----------+--------+-----------+----------+
