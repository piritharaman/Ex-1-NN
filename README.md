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

df= pd.read_csv("Churn_Modelling.csv")
print(df)

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)

df.duplicated()
print(df['EstimatedSalary'].describe())

scaler=MinMaxScaler()
df1 = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes(include='number')),
    columns=df.select_dtypes(include='number').columns
)
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))
print(X_test)
print("Lenght of X_test ",len(X_test))
```

## OUTPUT:
### Dataset
<img width="925" height="735" alt="image" src="https://github.com/user-attachments/assets/126c7b4e-afb1-47f9-bdbc-c1af0ca2cb5e" />

### X values
<img width="923" height="157" alt="image" src="https://github.com/user-attachments/assets/d904b45f-a856-4812-a330-63aa91a90195" />

### Y values
<img width="937" height="67" alt="image" src="https://github.com/user-attachments/assets/f0cae57c-e627-4ae7-86fc-2045500abfe0" />

### Null values
<img width="917" height="257" alt="image" src="https://github.com/user-attachments/assets/03a7db78-ec1e-4610-ae78-0c38377da415" />

<img width="926" height="317" alt="image" src="https://github.com/user-attachments/assets/c1fe7e92-c08f-4b11-bcff-1425f2e5ce47" />


### Duplicated values and Description
<img width="922" height="177" alt="image" src="https://github.com/user-attachments/assets/3df8d7a4-facb-4029-a730-1377e5888881" />


### Training Data
<img width="916" height="495" alt="image" src="https://github.com/user-attachments/assets/d5331fc8-2f2f-4187-822b-7fca9909d1f6" />

### Testing data
<img width="932" height="342" alt="image" src="https://github.com/user-attachments/assets/82809df6-4847-42a0-a5f5-49bf2eaaf07e" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


