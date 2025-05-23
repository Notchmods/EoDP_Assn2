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
from sklearn.model_selection import train_test_split,cross_val_score,KFold
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

#Contain One Hot Encoded column names for extracting dataframe.
xCol=[]
#Create a new DataFrame with One Hot Encoded values of the independent variables and the dependent variable
OHE_both_df = []
                         
    
def add_col():
    #Add vehicle damage level and one hot encoding for atmospheric
    #conditions into the dataframe.
    global OHE_both_df
    OHE_both_df = full_merged_df[["ACCIDENT_NO", "ROAD_TYPE", "VEHICLE_CATEGORY","ACCIDENT_TYPE"]].copy()
    OHE_both_df=pd.get_dummies(OHE_both_df,columns=["ROAD_TYPE","VEHICLE_CATEGORY","ACCIDENT_TYPE"])
    OHE_both_df[["FOG/SMOKE/DUST","RAIN/SNOW"]]=full_merged_df[["FOG/SMOKE/DUST","RAIN/SNOW"]]
    OHE_both_df["AVERAGE_INJ_LEVEL"]=full_merged_df["AVERAGE_INJ_LEVEL"]
        
    
    
    
"""Analyse Vehicle and environmental characteristics effects on vehicle damage and average injury level separately"""

#Train and predict vehicle damage based on vehicle and environmental factor together 
def both_analysis():
    #Merge both list together for the independent variable
    
    global xCol,OHE_both_df
    xCol=[col for col in OHE_both_df.columns if col!="ACCIDENT_NO"]
    #Add the dependent variable into the dataframe
    x=OHE_both_df[xCol]
    y=OHE_both_df["AVERAGE_INJ_LEVEL"]
    
    subtitle="Predicted vehicle damage level vs average injury level"
    organise_training_data(x,y,subtitle)
     
    
#Predict the frequency of accidents with both KNN and Logistic regression models.
def freq_accidents():
    #Count accident frequency by the attributes
    accident_freq_df = full_merged_df.groupby([
    'ROAD_TYPE', 
    'VEHICLE_CATEGORY',
    'MAIN_ATMOSPH_COND'
    ]).size().reset_index(name='ACCIDENT_COUNT')
       
    #Everything in the x is independent variable except for Accident count
    x = pd.get_dummies(accident_freq_df.drop(columns=['ACCIDENT_COUNT']))
    #Accident count is the dependent variable
    y = accident_freq_df['ACCIDENT_COUNT']
    #Organise train test split for data
    xTrain,xTest,yTrain,yTest= train_test_split(x,round(y),test_size=0.2,random_state=0)
    KNN_Continuous(x,y,xTrain,xTest,yTrain,yTest,accident_freq_df)
    DecisionTree_Continuous(xTrain,xTest,yTrain,yTest,accident_freq_df)

"""Machine learning part"""

    
#Organise a train test split for the data and run the 2 chosen Models
def organise_training_data(x,y,subtitle):
    xTrain,xTest,yTrain,yTest= train_test_split(x,round(y),test_size=0.1,random_state=0)
    #Apply 2 supervised learning models to the data set
    ApplyKNN(xTrain,xTest,yTrain,yTest,subtitle)
    DecisionTree(xTrain,xTest,yTrain,yTest,subtitle)
    

def ApplyKNN(xTrain,xTest,yTrain,yTest,subtitle):
    #Cross validation
    """
    # Define the range of k values to test
    k_values = range(1, 31)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Store the cross-validation scores for each k
    cv_scores = []

    for k in k_values:
        #Test different k's for KNN
        knn_model=KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_model, xTrain, yTrain, cv=kf, scoring='accuracy').mean()
        
    best_k = k_values[np.argmax(scores)]
    print(f"The optimal number of neighbors (k) is: {best_k}")

    # Display cross-validation scores for each k
    for k, score in zip(k_values, cv_scores):
        print(f"K: {k}, Cross-validation score: {score}")
    """

    #Training data
    knn_model=KNeighborsClassifier(n_neighbors=24)
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
    knn_reg = KNeighborsRegressor(n_neighbors=24, weights='distance')
    #Train the data
    knn_reg.fit(X_train_scaled, yTrain)
    #Testing the model by predicting the test data set.
    prediction = knn_reg.predict(X_test_scaled)
    print(prediction)
        
    print("Mean Squared Error:", mean_squared_error(yTest, prediction))
    print("R² Score:", r2_score(yTest, prediction))

    #Create table to display prediction
    decoded = accident_freq_df.iloc[xTest.index].copy()
    decoded['Predicted_ACCIDENT_COUNT'] = prediction

    print("\nSample prediction breakdown KNN:")
    print(decoded)

def DecisionTree(xTrain,xTest,yTrain,yTest,subtitle):
    #Cross validation
    """
    # Store all CV scores in a list
    cv_scores = []
    depth_range = range(1, 31)
    #Display cross-validation scores for each depth

    for depth in range(1, 31):  
        tree = DecisionTreeClassifier(max_depth=depth,random_state=42)  
        scores = cross_val_score(tree, xTrain, yTrain, cv=5, scoring='accuracy').mean()
        cv_scores.append(scores)
        print(f"Depth {depth}: CV Accuracy = {scores:.4f}")


    # Display cross-validation scores for each depth
    best_depth = depth_range[cv_scores.index(max(cv_scores))]
    print(f"\n Best Depth = {best_depth} with CV Accuracy = {max(cv_scores):.4f}")
"""
    #Train data
    #Set decision tree depth to 4
    tree= DecisionTreeClassifier(max_depth=4,random_state=42)
    #Train the data
    tree.fit(xTrain,yTrain)
    #Testing the model by predicting the test data set.
    prediction=tree.predict(xTest)
    #Predict and evaluate
    print("Decision Tree: ")
    print("Prediction: ",prediction)
    print("Accuracy:", accuracy_score(yTest, prediction))
    print(classification_report(yTest,prediction))
    #Confusion Matrix
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
      
    
   
def DecisionTree_Continuous(xTrain,xTest,yTrain,yTest,accident_freq_df):
    #Set decision tree depth to 7
    tree= DecisionTreeRegressor(max_depth=20,random_state=42)
    #Train the data
    tree.fit(xTrain,yTrain)
    #Testing the model by predicting the test data set.
    prediction=tree.predict(xTest)

    #Predict and evaluate
    print("Decision Tree: ")
    print("Prediction: ",prediction)
    print("Mean Squared Error:", mean_squared_error(yTest, prediction))
    print("R² Score:", r2_score(yTest, prediction))
    
    #Create table to display prediction
    decoded = accident_freq_df.iloc[xTest.index].copy()
    decoded['Predicted_ACCIDENT_COUNT'] = prediction
    print("\nSample prediction breakdown Decision Tree:")
    print(decoded)
    

    
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
    #Confusion matrix textual display
    print(cm)


#Execute the needed functions to get the result
add_col()
both_analysis()
freq_accidents()



