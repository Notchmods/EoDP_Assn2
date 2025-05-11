"""Question: What are the differences in traffic accident outcomes
among various vehicle types under different road and environmental conditions?

Independent variable-Environmental condition,vehicle types
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
full_merged_df=preprocessed.everything_df(
    "vehicle.csv", "accident.csv","atmospheric_cond.csv", "person.csv")
pd.set_option('display.max_columns', None)

full_merged_df=full_merged_df.head(10000)
#Specify features to look for
xOriginalCol=["ROAD_GEOMETRY_DESC","ROAD_TYPE","ROAD_SURFACE_TYPE_DESC",
           "ATMOSPH_COND","VEHICLE_CATEGORY"] #independent variable
yCol=["VEHICLE_DAMAGE_LEVEL","AVERAGE_INJ_LEVEL"] #dependent variable

xCol=[]


#Create a new DataFrame with One Hot Encoded values of the independent variables and the dependent variable
#Environmental factors
OHE_env_df = pd.get_dummies(full_merged_df[['ROAD_GEOMETRY_DESC', 'ROAD_TYPE', 'ROAD_SURFACE_TYPE_DESC']])

#vehicle type
OHE_vtype_df = pd.get_dummies(full_merged_df[['VEHICLE_CATEGORY']])

#both
OHE_both_df = pd.get_dummies(full_merged_df[['ROAD_GEOMETRY_DESC', 'ROAD_TYPE', 'ROAD_SURFACE_TYPE_DESC','VEHICLE_CATEGORY']])
                         

def extract_col():
    global xCol, xCol_vehicle,full_merged_df
    #Add extra columns before extracting column names
    add_col()
    #Extract all the columns of the independent variable from One Hot Encoded dataframe
    xCol = [col for col in OHE_env_df.columns]
    xCol_vehicle=[col for col in OHE_vtype_df.columns]
    

def add_col():
    #Add vehicle damage level and one hot encoding for atmospheric
    #conditions into the dataframe.
    OHE_env_df["VEHICLE_DAMAGE_LEVEL"]=full_merged_df["VEHICLE_DAMAGE_LEVEL"]
    OHE_env_df[["ATMOSPH_COND_DESC_Clear",
                "ATMOSPH_COND_DESC_Dust",
                "ATMOSPH_COND_DESC_Fog",
                "ATMOSPH_COND_DESC_Raining",
                "ATMOSPH_COND_DESC_Smoke",
                "ATMOSPH_COND_DESC_Snowing",
                "ATMOSPH_COND_DESC_Strong winds"]]=full_merged_df[["ATMOSPH_COND_DESC_Clear",
                "ATMOSPH_COND_DESC_Dust",
                "ATMOSPH_COND_DESC_Fog",
                "ATMOSPH_COND_DESC_Raining",
                "ATMOSPH_COND_DESC_Smoke",
                "ATMOSPH_COND_DESC_Snowing",
                "ATMOSPH_COND_DESC_Strong winds"]]

    OHE_both_df[["ATMOSPH_COND_DESC_Clear",
                "ATMOSPH_COND_DESC_Dust",
                "ATMOSPH_COND_DESC_Fog",
                "ATMOSPH_COND_DESC_Raining",
                "ATMOSPH_COND_DESC_Smoke",
                "ATMOSPH_COND_DESC_Snowing",
                "ATMOSPH_COND_DESC_Strong winds"]]=full_merged_df[["ATMOSPH_COND_DESC_Clear",
                "ATMOSPH_COND_DESC_Dust",
                "ATMOSPH_COND_DESC_Fog",
                "ATMOSPH_COND_DESC_Raining",
                "ATMOSPH_COND_DESC_Smoke",
                "ATMOSPH_COND_DESC_Snowing",
                "ATMOSPH_COND_DESC_Strong winds"]]


#Train and predict vehicle damage based on vehicle and environmental factor separately 
def VD_Analysis():
    #Environmental_factor_VD()
    Vehicle_type_factor_VD()

     
#Train and predict Average Injury Level based on vehicle and environmental factor separately 
def AIL_Analysis():
    #Environmental_factor_AIL()
    Vehicle_type_factor_AIL()

#Train and predict vehicle damage based on vehicle and environmental factor together 
def both_analysis():
    #Merge both list together for the independent variable
    x_new_Col=xCol+xCol_vehicle
    OHE_both_df["VEHICLE_DAMAGE_LEVEL"]=full_merged_df["VEHICLE_DAMAGE_LEVEL"]
    x=OHE_both_df[x_new_Col]
    y=OHE_both_df ["VEHICLE_DAMAGE_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted vehicle damage level vs actual vehicle damage level"
    organise_training_data(x,y,title)
    


#Analyse whether Environmental factors will have an impact on vehicle damage
def Environmental_factor_VD():
    #Prepare independent and dependent variable for training
    OHE_env_df["VEHICLE_DAMAGE_LEVEL"]=full_merged_df["VEHICLE_DAMAGE_LEVEL"]
    x=OHE_env_df[xCol]
    y=OHE_env_df["VEHICLE_DAMAGE_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted vehicle damage level level vs actual vehicle damage level"
    organise_training_data(x,y,title)
    

#Analyse whether vehicle type will have an impact on vehicle damage
def Vehicle_type_factor_VD():
    #Prepare independent and dependent variable for training
    OHE_vtype_df["VEHICLE_DAMAGE_LEVEL"]=full_merged_df["VEHICLE_DAMAGE_LEVEL"]
    x=OHE_vtype_df[xCol_vehicle]
    y=OHE_vtype_df["VEHICLE_DAMAGE_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted vehicle damage level vs actual vehicle damage level"
    organise_training_data(x,y,title)


#Analyse whether Environmental factors will have an impact on Average Injury Level
def Environmental_factor_AIL():
    #Prepare independent and dependent variable for training
    OHE_env_df["AVERAGE_INJ_LEVEL"]=full_merged_df["AVERAGE_INJ_LEVEL"]
    x=OHE_env_df[xCol]
    y=OHE_env_df["AVERAGE_INJ_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted average injury level vs actual average injury level"
    organise_training_data(x,y,title)
   

#Analyse whether vehicle type will have an impact on Average Injury Level
def Vehicle_type_factor_AIL():
    #Prepare independent and dependent variable for training
    OHE_vtype_df["AVERAGE_INJ_LEVEL"]=full_merged_df["AVERAGE_INJ_LEVEL"]
    x=OHE_vtype_df[xCol_vehicle]
    y=OHE_vtype_df["AVERAGE_INJ_LEVEL"]
    #Split into testing and training data for vehicle damage level
    title="Predicted vehicle damage level vs actual vehicle damage level"
    organise_training_data(x,y,title)

"""Machine learning part"""

    
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
    knn_model=KNeighborsClassifier(metric='manhattan',n_neighbors=10)
    
    
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

extract_col()
both_analysis()


