## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```python
import pandas as pd
df = pd.read_csv("Encoding Data.csv")
df
```

<img width="641" height="478" alt="image" src="https://github.com/user-attachments/assets/0c2df576-2367-406d-95eb-fc7f477431e7" />

```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="584" height="237" alt="{C9F77E47-A215-40B6-A933-576D0E719E22}" src="https://github.com/user-attachments/assets/45e4038b-385a-4693-8c22-083c8fcece07" />

```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="866" height="453" alt="{CDAE8B83-8013-4764-8412-1529C193EC55}" src="https://github.com/user-attachments/assets/099e412c-3ba6-4e93-ad43-e3d511d757c5" />

```python
le=LabelEncoder()
df1=df.copy()
df1['ord_2']=le.fit_transform(dfc['ord_2'])
df1
```

<img width="881" height="460" alt="image" src="https://github.com/user-attachments/assets/271cf689-f8c6-4d57-8545-18e53147eeb1" />

```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="847" height="456" alt="{7F0F6647-9555-478B-9BCB-1DD29DB7B1A6}" src="https://github.com/user-attachments/assets/1fa17df2-8323-4051-bef1-55ae65265144" />

```python
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="1016" height="444" alt="{05382C01-8AF1-467B-8C37-9B9985E5FCB6}" src="https://github.com/user-attachments/assets/6b2df9f2-fa43-4a95-bc3d-eb997d76b90c" />

```python
pip install --upgrade category_encoders
```

<img width="1771" height="319" alt="{D088B71E-B13D-4433-8346-2A83788D2BCB}" src="https://github.com/user-attachments/assets/5f88a867-980a-4674-9dec-2be112ab7a0f" />

```python
pip list
```

<img width="1751" height="653" alt="{7F52E6CD-1D9E-49EE-B394-725155631052}" src="https://github.com/user-attachments/assets/89952d55-14fe-4178-b780-cf37834bbd34" />

```python
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

<img width="1035" height="463" alt="{8C8DE783-6417-4E61-B27C-5457C53641AC}" src="https://github.com/user-attachments/assets/9e613d7d-b8e5-46a7-a046-1afdedf079e8" />

```python
bi=BinaryEncoder()
a=bi.fit_transform(df['Ord_2'])
df1=pd.concat([df,a],axis=1)
df1
```

<img width="1283" height="449" alt="{C29B121A-E4E3-488C-B700-7463057D5F17}" src="https://github.com/user-attachments/assets/ac62db97-207b-48c0-9696-c4d3d49e97dc" />

```python
from category_encoders import TargetEncoder
ta=TargetEncoder()
Cpp=df.copy()
new=ta.fit_transform(X=Cpp["City"],y=Cpp["Target"])
Cpp=pd.concat([Cpp,new],axis=1)
Cpp
```
<img width="1085" height="454" alt="{8519E392-DD18-4416-8995-9C7BE3E6E7A0}" src="https://github.com/user-attachments/assets/1d5bbddb-f2bc-4753-86cf-2e63d0629d10" />

```python
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1408" height="531" alt="{F55D149A-8550-470D-9EF6-146194799F42}" src="https://github.com/user-attachments/assets/0ad671b8-7909-451d-890c-582436576666" />

```python
df.skew()
```


<img width="673" height="226" alt="{744E66E0-9C1D-4822-B80A-4078D99751CB}" src="https://github.com/user-attachments/assets/88cb200c-297f-4aee-bb04-8bda0ff3b2cf" />

```python
np.log(df["Highly Positive Skew"])
```

<img width="651" height="544" alt="{3D10D997-BDBF-4AA4-9368-6961B5684210}" src="https://github.com/user-attachments/assets/1cefc62d-5819-4897-b9f5-da87609cc093" />


```python
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="653" height="519" alt="{7B03A147-BB5B-42E5-9626-0CA382AEC45D}" src="https://github.com/user-attachments/assets/bcd21c41-f9b3-48d1-beff-656ca1130a47" />

```python
np.sqrt(df["Highly Positive Skew"])
```

<img width="603" height="529" alt="{B79E210E-DE17-4AFA-8D6E-93C3C766B898}" src="https://github.com/user-attachments/assets/daf979d1-b0f9-482c-a2de-616484a054a8" />

```python
np.square(df["Highly Positive Skew"])
```

<img width="712" height="532" alt="{6DC40495-2260-4373-A724-E87AD1A23A73}" src="https://github.com/user-attachments/assets/10db3af8-5c5a-4740-a950-3afc6154a772" />

```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1598" height="520" alt="{214D3856-C150-4F7B-8AA8-BA99B770AA9F}" src="https://github.com/user-attachments/assets/e58be466-862c-4582-b291-c7340504be80" />

```python
df.skew()
```

<img width="583" height="287" alt="image" src="https://github.com/user-attachments/assets/b9596967-f62e-40a3-aa0b-278d459de239" />

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="733" height="335" alt="{CBF95CB4-3B76-48E1-8B6E-93DC9038B85C}" src="https://github.com/user-attachments/assets/e5a567ff-8990-4227-916c-40502d260881" />

```python
 from sklearn.preprocessing import QuantileTransformer
 qat=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qat.fit_transform(df[["Moderate Negative Skew"]])
 df
```

<img width="1779" height="565" alt="{3930C7D4-EF33-4792-89D6-C48E1886929C}" src="https://github.com/user-attachments/assets/300ad857-418d-457c-b6c2-267208fa6e03" />

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="951" height="563" alt="{60AF097C-4AC7-4576-A442-10AE7C8E2D0E}" src="https://github.com/user-attachments/assets/265a9aa2-a923-453f-9592-4a615caef213" />

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="920" height="544" alt="{2163B69F-472C-4CFD-9A22-95C62746133D}" src="https://github.com/user-attachments/assets/12d4dbd2-710f-4855-9f8a-dde71a3afc9f" />

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="1062" height="551" alt="{A41709BC-367C-4574-B438-DA18F2617DAF}" src="https://github.com/user-attachments/assets/7f7fb361-80bf-44d7-9c2b-16fb5f36afa5" />

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="948" height="575" alt="{8416C41F-EB7E-434A-B2CA-472284F94765}" src="https://github.com/user-attachments/assets/62d15f5b-d734-4973-ae79-2216e9ee6333" />

```python
dt=pd.read_csv("data.csv")
dt
```

<img width="1085" height="461" alt="{76842519-E545-47E7-892A-ACEB6CF885B0}" src="https://github.com/user-attachments/assets/0f06a1c5-b771-4619-8e38-8480578c6f58" />

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Ord_1"]=qt.fit_transform(dt[["Target"]])
sm.qqplot(dt['Target'],line='45')
plt.show()
```

<img width="1705" height="590" alt="{5E8D12E7-1AF8-49E0-A581-B064A0856652}" src="https://github.com/user-attachments/assets/5b6ac2d0-2c0b-4c5e-8c66-5d9054d0735f" />

```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="1092" height="551" alt="{0950DE37-C237-4C40-B4E5-8B7667D1BE12}" src="https://github.com/user-attachments/assets/5518517b-6b7f-43fa-b5e8-a36330b65073" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successf


       
