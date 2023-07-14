
import pandas as pd
import re
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


data = pd.read_csv("cleaned_disease_predection.csv")
description = pd.read_csv('symptom_Description.csv')
precautions = pd.read_csv('symptom_precaution.csv')
#data = open('cleaned_disease_predection.csv', errors='ignore')

columns_present_in_dataset = []
for col in data.columns:
    for c in col:
        s1=re.sub("[_]"," ",c)
        columns_present_in_dataset.append(s1)

#data=data.drop(['Unnamed: 133'], axis=1)
x=data.drop(["prognosis"],axis=1)
encoder = LabelEncoder()
y= encoder.fit_transform(data["prognosis"])
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2, random_state =0)


nb= GaussianNB()
nb.fit(x_train,y_train)
scores = cross_val_score(nb, x, y, cv = 10,
                         n_jobs = -1)
y_test_pred_nb = nb.predict(x_test)
y_train_pred_nb = nb.predict(x_train)
Acc_y_train_nb = accuracy_score(y_train, nb.predict(x_train))*100
Acc_y_test_nb = accuracy_score(y_test, y_test_pred_nb)*100
conf_mat_nb = confusion_matrix(y_test, y_test_pred_nb)
#print(f"Scores: {scores}")
#print(f"Mean Score: {np.mean(scores)}")


rfc= RandomForestClassifier()
rfc.fit(x_train,y_train)
scores = cross_val_score(rfc, x, y, cv = 10,
                         n_jobs = -1)
y_test_pred_rfc = rfc.predict(x_test)
y_train_pred_rfc = rfc.predict(x_train)
Acc_y_train_rfc = accuracy_score(y_train, rfc.predict(x_train))*100
Acc_y_test_rfc = accuracy_score(y_test, y_test_pred_rfc)*100
conf_mat_rfc = confusion_matrix(y_test, y_test_pred_rfc)

dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
scores = cross_val_score(dtc, x, y, cv = 10,
                         n_jobs = -1)
y_test_pred_dtc = dtc.predict(x_test)
y_train_pred_dtc = dtc.predict(x_train)
Acc_y_train_dtc= accuracy_score(y_train, dtc.predict(x_train))*100
Acc_y_test_dtc = accuracy_score(y_test, y_test_pred_dtc)*100
conf_mat_dtc = confusion_matrix(y_test, y_test_pred_dtc)

symptoms = x.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
data_columns = []
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
for i in data.columns:
    if i not in data_columns:
        v = " ".join([j.capitalize() for j in i.split("_")])
        data_columns.append(v)

def symptoms_predections(text):
    space_removed = text.replace("and",",")
    filtered_sent = ''
    for i in data_columns:
        if i in space_removed:
            filtered_sent += f'{i},'

    filt_symptoms = filtered_sent.split(",")

    filter_symptoms = filt_symptoms.pop(len(filt_symptoms)-1)

    input_data =  [0] * len(data_dict["symptom_index"])
    for s in filt_symptoms:
        index = data_dict["symptom_index"][s]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1,-1)
    rfc_prediction = data_dict["predictions_classes"][rfc.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb.predict(input_data)[0]]
    dtc_prediction = data_dict["predictions_classes"][dtc.predict(input_data)[0]]
    global final_prediction
    final_prediction = mode([rfc_prediction, nb_prediction,dtc_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rfc_prediction,
        "naive_bayes_prediction" : nb_prediction,
        "Decision_Tree_prediction" : dtc_prediction,
        "final_prediction":final_prediction
    }
    return final_prediction
def chatbot_response(Usertext):
    text = ['Yes','YES','yes']
    text1 = ['prevent','Prevent','precaution','Precaution']
    text3 =['Yes I want to talk with Doctor','I want to consult with Doctor','yes i want to talk with Doctor','I Want To Consult With Doctor']
    text4 = ['@']
    text5 = ['Bye','Thankyou','bye bye','bye','thankyou']
    for t in text:
        if t == Usertext:
            Diseases = description['Disease']
            Description = description['Description']
            for i in range(len(description)):
                if final_prediction == Diseases[i]:
                    print(Diseases[i])
                    print(Description[i])
                    return (f" {Description[i]}. Type 'Precaution' for more information on how to prevent this. ")
    for t in text1:
        if t == Usertext:
            prec_dis = precautions['Disease']
            prec1 = precautions['Precaution_1']
            prec2 = precautions['Precaution_2']
            prec3 = precautions['Precaution_3']
            prec4 = precautions['Precaution_4']
            for i in range(len(precautions)):
                if final_prediction == prec_dis[i]:
                    return (f"You can follow these tips to prevent {final_prediction}, such as {prec1[i]}, {prec2[i]}, {prec3[i]}, {prec4[i]}. Do you want to consult with the doctor?")
    for t in text3:
        if t in Usertext:
            return "ok great! Please share your email address with me so that I can schedule an appointment with the doctor as shortly as possible. "
    for t in text4:
        if t in Usertext:
            return "I've sent you a link. Please consult with the doctor. Thankyou"
    greets = ["hello","hey","hi","hi bot","hello bot","Hello bot"]
    exixts = ["ok Thankyou","Thankyou","Bye","ok","Thankyou so much","thankyou"]
    for g in greets:
        if Usertext == g:
            user_greeted = ("Hi, I can help you if you're feeling unwell. Please describe your symptoms so that I can identify them")
            return user_greeted

    if len(Usertext) > 5:
        return (f"It seems you have {symptoms_predections(Usertext)}.Type 'Yes' if you want to know more about {symptoms_predections(Usertext)}.")

    for g in exixts:
        if Usertext == g:
            return ("Thankyou")

from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()

#
#%%



