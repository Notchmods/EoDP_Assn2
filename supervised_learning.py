"""Question: What are the differences in traffic accident outcomes
among various vehicle types under different road and environmental conditions?

Independent variable-Environmental condition
Dependent variable- Vehicle damage and Average Injury level


The Supervised learning algorithm that I've selected are
K-Nearest Neighbor and Logistic Regression

"""

#Import all the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder    
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pre as preprocessed

#Import the pre-processed data from the pre.py file
full_merged_df=preprocessed.everything_df("vehicle.csv", "accident.csv","atmospheric_cond.csv", "person.csv").head(1000)
pd.set_option('display.max_columns', None)

#Specify features to look for
xOriginalCol=["ROAD_GEOMETRY_DESC","ROAD_TYPE","ROAD_SURFACE_TYPE_DESC",
           "ATMOSPH_COND"] #independent variable
yCol=["VEHICLE_DAMAGE_LEVEL","AVERAGE_INJ_LEVEL"] #dependent variable


#Create a new DataFrame with One Hot Encoded values of the independent variables and the dependent variable
OHE_df = pd.get_dummies(full_merged_df[['ROAD_GEOMETRY_DESC', 'ROAD_TYPE', 'ROAD_SURFACE_TYPE_DESC','ATMOSPH_COND']])

#Extract all the columns of the independent variable from One Hot Encoded dataframe
xCol = [col for col in OHE_df.columns]
   
#Analyse vehicle damage 
def VD_Analysis():
    #Prepare independent and dependent variable for training
    OHE_df["VEHICLE_DAMAGE_LEVEL"]=full_merged_df["VEHICLE_DAMAGE_LEVEL"]
    x=OHE_df[xCol]
    y=OHE_df["VEHICLE_DAMAGE_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted vehicle damage level injury level vs actual injury level"
    organise_training_data(x,y,title)
     

#Analyse Average Injury Level
def AIL_Analysis():
    #Prepare independent and dependent variable for training
    OHE_df["AVERAGE_INJ_LEVEL"]=full_merged_df["AVERAGE_INJ_LEVEL"]
    x=OHE_df[xCol]
    y=OHE_df["AVERAGE_INJ_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted average injury level vs actual injury level"
    organise_training_data(x,y,title)


    
#Organise a train test split for the data and run the 2 chosen Models
def organise_training_data(x,y,title):
    xTrain,xTest,yTrain,yTest= train_test_split(x,round(y),test_size=0.2,random_state=0)
    #Apply 2 supervised learning models to the data set
    ApplyKNN(xTrain,xTest,yTrain,yTest,title)
    ApplyLogReg(xTest,yTest,xTrain,yTrain,title)
    

def ApplyKNN(xTrain,xTest,yTrain,yTest,title):

    #Scale the KNN model
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(xTrain)
    x_test_scaled = scaler.transform(xTest)
    
    #Create a KNN model
    knn_model=KNeighborsClassifier(weights='distance',n_neighbors=10)
    
    #Fit the model with class weight and train it
    knn_model.fit(x_train_scaled,yTrain)
    #Create prediction for the dependent variable
    prediction= knn_model.predict(x_test_scaled)

    #Predict and evaluate
    print("KNN: ")
    print("Prediction: ",prediction)
    print("Accuracy:", accuracy_score(yTest, prediction))
    print(classification_report(yTest,prediction))
    ConfusionMatrix(xTest,yTest,xTrain,yTrain,prediction,title)



def ApplyLogReg(xTest,yTest,xTrain,yTrain,title):
    #Scale the x for logistic regression model
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(xTrain)
    x_test_scaled = scaler.transform(xTest)
    
    #Logistic regression
    logreg= LogisticRegression(class_weight='balanced',random_state=16)
    logreg.fit(x_train_scaled,yTrain)
    prediction=logreg.predict(x_test_scaled)

    #Predict and evaluate result
    print("Logistic Regression: ")
    print("Prediction: ",prediction)
    print("Accuracy:", accuracy_score(yTest, prediction))
    print(classification_report(yTest,prediction))
    ConfusionMatrix(xTest,yTest,xTrain,yTrain,prediction,title)


    
#Evaluate the model with confusion matrix
def ConfusionMatrix(xTest,yTest,xTrain,yTrain,prediction,title):
    #Plot confusion matrix
    cm=confusion_matrix(yTest, prediction)
    display= ConfusionMatrixDisplay(confusion_matrix=cm)
    #Represent confusion matrix as a heatmap
    display.plot(cmap=plt.cm.Blues)
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.show()
    print(cm)

VD_Analysis()
AIL_Analysis()




