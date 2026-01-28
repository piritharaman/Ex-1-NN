<H3>NAME: PIRITHARAMAN R</H3>
<H3>REGISTER NO: 212223230148</H3>
<H3>EX.NO.1</H3>
<H3>DATE: 28/01/26</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("Churn_Modelling.csv")
print(data.head())
print(data.tail())
X=data.iloc[:,:-1].values
print(X)
y=data.iloc[:,-1].values
print(y)
data.info()
print("Missing Values: \n ",data.isnull().sum())
print("Duplicate values:\n ")
print(data.duplicated())
data.describe()
data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
X = data.drop('Exited', axis=1)  
y = data['Exited'] 
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training data")
print(X_train)
print(y_train)
print("Testing data")
print(X_test)
print(y_test)
print("Length of X_test: ", len(X_test))
```

## OUTPUT:
### Dataset
<img width="669" height="225" alt="image" src="https://github.com/user-attachments/assets/ca7255a7-537e-4e3c-91ea-a78decfa5468" />

### X values
<img width="379" height="128" alt="image" src="https://github.com/user-attachments/assets/463e4da8-5aac-41d0-830a-777aaee50b60" />

### Y values
<img width="142" height="26" alt="image" src="https://github.com/user-attachments/assets/4dfd1672-62c1-4963-bf4f-4b24d673210d" />

### Null values
<img width="217" height="282" alt="image" src="https://github.com/user-attachments/assets/0916117b-3167-440f-b10f-dcff40d1387f" />

### Duplicated values
<img width="245" height="240" alt="image" src="https://github.com/user-attachments/assets/aa741c53-21e7-4f7d-8999-83310fe5aab5" />

### Description
<img width="619" height="408" alt="image" src="https://github.com/user-attachments/assets/9fba2902-3875-41c5-980f-c3b5072a0334" />

### Training Data
<img width="549" height="327" alt="image" src="https://github.com/user-attachments/assets/7afc6eae-45bb-431a-9de9-e05418114cda" />

### Testing data
<img width="535" height="329" alt="image" src="https://github.com/user-attachments/assets/9c41e124-4d44-4eed-9e62-02b022ed2733" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


