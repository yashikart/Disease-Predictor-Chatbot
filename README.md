# SyncInterns_Task1_MachineLearningInterns
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ChatBot that can Predict Disease
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
I build a Simple chatbot with the topic of health care. With the help of user-provided symptoms, this bot can identify several diseases. The bot will also provide advice on how to prevent this disease. Additionally, the bot asks the user if they want to consult with a doctor; if so, the user must provide their mail address so that the bot can send them an appointment link.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Software Used :  Anaconda, Jupyter, DataSpell
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Web Framework : Flask
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Programming Language : Python
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Libraries : Pandas, Numpy, nltk, scikit-learn, matplotlib, seaborn
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Dataset  : Kaggle (Traning.csv) & (Testing.csv)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Algorithm : Gaussian Naïve Bayes, Random Forest Classification, Decision Tree Classifier 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The following are some basic intends I've added to the chatbots:
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    text = ['Yes','YES','yes']
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    text1 = ['prevent','Prevent','precaution','Precaution']
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    text3 =['Yes I want to talk with Doctor','I want to consult with Doctor','yes i want to talk with Doctor','I Want To Consult With Doctor']
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    text4 = ['@']
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    text5 = ['Bye','Thankyou','bye bye','bye','thankyou']
 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    for Disease Description Intents - I have used Kaggel Dataset (symptom_Description.csv)
 --------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    for Disease Precaution Intents - I have used Kaggel Dataset (symptom_precaution.csv)
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------****Note***
 
    Symptoms' starting letters must be written in uppercase letters. 
     for eg: Loss Of Apetite, Cough, High Fever, Abdominal Pain.
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
     Symptoms words should be entered exactly as they are shown in the Traning.csv columns.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

['Unnamed: 0', 'Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering', 'Chills', 'Joint Pain', 'Stomach Pain', 'Acidity', 'Ulcers On Tongue', 'Muscle Wasting', 'Vomiting', 'Burning Micturition', 'Spotting  urination', 'Fatigue', 'Weight Gain', 'Anxiety', 'Cold Hands And Feets', 'Mood Swings', 'Weight Loss', 'Restlessness', 'Lethargy', 'Patches In Throat', 'Irregular Sugar Level', 'Cough', 'High Fever', 'Sunken Eyes', 'Breathlessness', 'Sweating', 'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin', 'Dark Urine', 'Nausea', 'Loss Of Appetite', 'Pain Behind The Eyes', 'Back Pain', 'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever', 'Yellow Urine', 'Yellowing Of Eyes', 'Acute Liver Failure', 'Fluid Overload', 'Swelling Of Stomach', 'Swelled Lymph Nodes', 'Malaise', 'Blurred And Distorted Vision', 'Phlegm', 'Throat Irritation', 'Redness Of Eyes', 'Sinus Pressure', 'Runny Nose', 'Congestion', 'Chest Pain', 'Weakness In Limbs', 'Fast Heart Rate', 'Pain During Bowel Movements', 'Pain In Anal Region', 'Bloody Stool', 'Irritation In Anus', 'Neck Pain', 'Dizziness', 'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels', 'Puffy Face And Eyes', 'Enlarged Thyroid', 'Brittle Nails', 'Swollen Extremeties', 'Excessive Hunger', 'Extra Marital Contacts', 'Drying And Tingling Lips', 'Slurred Speech', 'Knee Pain', 'Hip Joint Pain', 'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness', 'Spinning Movements', 'Loss Of Balance', 'Unsteadiness', 'Weakness Of One Body Side', 'Loss Of Smell', 'Bladder Discomfort', 'Foul Smell Of urine', 'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching', 'Toxic Look (typhos)', 'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium', 'Red Spots Over Body', 'Belly Pain', 'Abnormal Menstruation', 'Dischromic  Patches', 'Watering From Eyes', 'Increased Appetite', 'Polyuria', 'Family History', 'Mucoid Sputum', 'Rusty Sputum', 'Lack Of Concentration', 'Visual Disturbances', 'Receiving Blood Transfusion', 'Receiving Unsterile Injections', 'Coma', 'Stomach Bleeding', 'Distention Of Abdomen', 'History Of Alcohol Consumption', 'Fluid Overload.1', 'Blood In Sputum', 'Prominent Veins On Calf', 'Palpitations', 'Painful Walking', 'Pus Filled Pimples', 'Blackheads', 'Scurring', 'Skin Peeling', 'Silver Like Dusting', 'Small Dents In Nails', 'Inflammatory Nails', 'Blister', 'Red Sore Around Nose', 'Yellow Crust Ooze', 'Prognosis']
