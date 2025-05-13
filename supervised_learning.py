"""Question: What are the differences in traffic accident outcomes
among various vehicle types under different road and environmental conditions?

Independent variable-Environmental condition,vehicle types
Dependent variable- Vehicle damage and Average Injury level


The Supervised learning algorithm that I've selected are
K-Nearest Neighbor and Decision Tree

"""

#Import all the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder    
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import export_text,DecisionTreeClassifier,DecisionTreeRegressor,plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pre as preprocessed

#Import the pre-processed data from the pre.py file
full_merged_df=preprocessed.everything_df(
    "vehicle.csv", "accident.csv","atmospheric_cond.csv", "person.csv")
pd.set_option('display.max_columns', None)
full_merged_df=full_merged_df.head(1000)    
#Specify features to look for
xOriginalCol=["ROAD_TYPE",
           "ATMOSPH_COND","VEHICLE_CATEGORY"] #independent variable
yCol=["VEHICLE_DAMAGE_LEVEL","AVERAGE_INJ_LEVEL"] #dependent variable

xCol=[]
#Create a new DataFrame with One Hot Encoded values of the independent variables and the dependent variable
OHE_both_df = []
                         
    
def add_col():
    #Add vehicle damage level and one hot encoding for atmospheric
    #conditions into the dataframe.
    global OHE_both_df
    OHE_both_df=pd.get_dummies(full_merged_df[['ROAD_TYPE','VEHICLE_CATEGORY']])
    OHE_both_df[["CLEAR","FOG/SMOKE/DUST","RAIN/SNOW"]]=full_merged_df[["CLEAR","FOG/SMOKE/DUST","RAIN/SNOW"]]
    
"""Analyse Vehicle and environmental characteristics effects on vehicle damage and average injury level separately"""

#Train and predict vehicle damage based on vehicle and environmental factor together 
def both_analysis():
    #Merge both list together for the independent variable
    global xCol
    xCol=[col for col in OHE_both_df.columns]
    
    #Add the dependent variable into the dataframe
    OHE_both_df["VEHICLE_DAMAGE_LEVEL"]=full_merged_df["VEHICLE_DAMAGE_LEVEL"]
    OHE_both_df["AVERAGE_INJ_LEVEL"]=full_merged_df["AVERAGE_INJ_LEVEL"]
    #Analyse vehicle damage level first.
    x=OHE_both_df[xCol]
    y=OHE_both_df["VEHICLE_DAMAGE_LEVEL"]
    #Split into testing and training data for vehicle damage level
    subtitle="Predicted vehicle damage level vs actual vehicle damage level"
    organise_training_data(x,y,subtitle)

    #Analyse Average Injury Level
    y=OHE_both_df["AVERAGE_INJ_LEVEL"]
    subtitle="Predicted vehicle damage level vs average injury level"
    organise_training_data(x,y,subtitle)
     
    
#Predict the frequency of accidents with both KNN and Logistic regression models.
def freq_accidents():
    #Count accident frequency by the attributes
    accident_freq_df = full_merged_df.groupby([
    'ROAD_TYPE', 
    'VEHICLE_CATEGORY'
    ]).size().reset_index(name='ACCIDENT_COUNT')
       
    #Everything in the x is independent variable except for Accident count
    x = pd.get_dummies(accident_freq_df.drop(columns='ACCIDENT_COUNT'))
    #Accident count is the dependent variable
    y = accident_freq_df['ACCIDENT_COUNT']
    #Organise train test split for data
    xTrain,xTest,yTrain,yTest= train_test_split(x,round(y),test_size=0.2,random_state=0)
    KNN_Continuous(x,y,xTrain,xTest,yTrain,yTest,accident_freq_df)
    DecisionTree_Continuous(xTrain,xTest,yTrain,yTest)

"""Machine learning part"""

    
#Organise a train test split for the data and run the 2 chosen Models
def organise_training_data(x,y,subtitle):
    xTrain,xTest,yTrain,yTest= train_test_split(x,round(y),test_size=0.5,random_state=0)
    #Apply 2 supervised learning models to the data set
    ApplyKNN(xTrain,xTest,yTrain,yTest,subtitle)
    DecisionTree(xTrain,xTest,yTrain,yTest,subtitle)
    

def ApplyKNN(xTrain,xTest,yTrain,yTest,subtitle):
    #Create a KNN model using Manhattan metric with k=5
    knn_model=KNeighborsClassifier(n_neighbors=5)
    
    #Fit the model with class weight and train it
    knn_model.fit(xTrain,yTrain)
    #Testing the model by predicting the test data set.
    prediction= knn_model.predict(xTest)    

    #Predict and evaluate
    print("KNN: ")
    print("Prediction: ",prediction)
    print("Accuracy:", accuracy_score(yTest, prediction))
    print(classification_report(yTest,prediction))
    ConfusionMatrix(xTest,yTest,xTrain,yTrain,prediction,subtitle)    

def KNN_Continuous(x,y,xTrain,xTest,yTrain,yTest,accident_freq_df):
    #Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(xTrain)
    X_test_scaled = scaler.transform(xTest)
    #Use KNN Regressor as the accident count is a continuous value rather than categorical
    knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
    #Train the data
    knn_reg.fit(X_train_scaled, yTrain)
    #Testing the model by predicting the test data set.
    prediction = knn_reg.predict(X_test_scaled)
    print(prediction)

    print("Mean Squared Error:", mean_squared_error(yTest, prediction))
    print("R² Score:", r2_score(yTest, prediction))
    decoded = accident_freq_df.iloc[xTest.index].copy()
    decoded['Actual_ACCIDENT_COUNT'] = yTest.values
    decoded['Predicted_ACCIDENT_COUNT'] = prediction

    print("\nSample prediction breakdown:")
    print(decoded)

def DecisionTree(xTrain,xTest,yTrain,yTest,subtitle):
    #Set decision tree depth to 5
    tree= DecisionTreeClassifier(max_depth=5,class_weight='balanced',random_state=42)
    #Train the data
    tree.fit(xTrain,yTrain)
    #Testing the model by predicting the test data set.
    prediction=tree.predict(xTest)
    #Predict and evaluate
    print("Decision Tree: ")
    print("Prediction: ",prediction)
    print("Accuracy:", accuracy_score(yTest, prediction))
    print(classification_report(yTest,prediction))
    ConfusionMatrix(xTest,yTest,xTrain,yTrain,prediction,subtitle)
    plot_tree(tree,
          feature_names=xTest.columns,     # show your actual feature names
          filled=True,                 # color-code nodes
          rounded=True,                # round corners of nodes
          fontsize=5)                 # size of the text inside the boxes
    tree_rules = export_text(tree, feature_names=list(xTest.columns))
    print(tree_rules)
    plt.title("Decision Tree - Accident Prediction", fontsize=16)
    plt.show()
        
    print(classification_report(yTest,prediction))
      
    
   
def DecisionTree_Continuous(xTrain,xTest,yTrain,yTest):
    #Set decision tree depth to 5
    tree= DecisionTreeRegressor(max_depth=5,random_state=42)
    #Train the data
    tree.fit(xTrain,yTrain)
    #Testing the model by predicting the test data set.
    prediction=tree.predict(xTest)

    #Predict and evaluate
    print("Decision Tree: ")
    print("Prediction: ",prediction)
    print("Mean Squared Error:", mean_squared_error(yTest, prediction))
    print("R² Score:", r2_score(yTest, prediction))


    
#Evaluate the model with confusion matrix
def ConfusionMatrix(xTest,yTest,xTrain,yTrain,prediction,subtitle):
    #Plot confusion matrix
    cm=confusion_matrix(yTest, prediction)
    display= ConfusionMatrixDisplay(confusion_matrix=cm)
    #Represent confusion matrix as a heatmap
    display.plot(cmap=plt.cm.Blues)
    plt.title(subtitle, fontsize=11, pad=20)
    plt.xlabel('Predicted', x=0.5, y=3,fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.show()
    print(cm)


#Execute the needed functions to get the result
add_col()
both_analysis()
freq_accidents()



