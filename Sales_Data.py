#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division
"""
Created on Sat Dec 14 20:56:23 2019

@author: jjs0sbw
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

import warnings
warnings.filterwarnings('ignore')

'''
###############################################################################
#####  Milestone 3 - Data Model And Evaluation
##### 
#####  Work Snohomish County Real Estate Data
#####
#####  Joseph Simpson - UW Data Science Fall 2019
#####
###############################################################################
#####
#####  Item 1: Preparation of Data Set
#####    Item 1.1:  Source citation for your data set
#####    Item 1.2:  Data read from an easily and freely accessible source
#####    Item 1.3:  Number of observations and attributes
#####    Item 1.4:  Data types
#####    Item 1.5:  Distribution of numerical variables
#####    Item 1.6:  Distribution of categorical variables
#####    Item 1.7:  A comment on each attribute
#####    Item 1.8:  Removing cases with missing data
#####    Item 1.9:  Removing outliers
#####    Item 1.10: Imputing missing values
#####    Item 1.11: Decoding
#####    Item 1.12: Consolidation
#####    Item 1.13: One-hot encoding
#####    Item 1.14: Normalization
#####
#####  Item 2: Unsupervised Learning
#####    Item 2.1:  Perform a K-Means with sklearn on some of your attributes.
#####               Include at least one categorical column and one numeric 
#####               attribute. Neither may be a proxy for the expert label 
#####               in supervised learning.
#####    Item 2.2   Normalize the attributes prior to K-Means or justify why 
#####               you didn't normalize.
#####    Item 2.3   Add the cluster label to the data set to be used 
#####               in supervised learning
#####
#####  Item 3: Supervised Learning
#####    Item 3.1: Ask a binary-choice question that describes your 
#####              classification. Write the question as a comment.
#####    Item 3.2: Split your data set into training and testing sets using 
#####              the proper function in sklearn.
#####    Item 3.3: Use sklearn to train two classifiers on your training set, 
#####              like logistic regression and random forest. 
#####    Item 3.4: Apply your (trained) classifiers to the test set.
#####    Item 3.5: Create and present a confusion matrix for each classifier. 
#####    Item 3.6: Specify and justify your choice of probability threshold.
#####    Item 3.7: For each classifier, create and present 2 accuracy metrics 
#####              based on the confusion matrix of the classifier.
#####    Item 3.8: For each classifier, calculate the ROC curve and it's AUC 
#####              using sklearn. Present the ROC curve. Present the AUC in 
#####              the ROC's plot.
#####
###############################################################################
'''

###############################################################################
#####  Item 1: Preparation of Data Set
###############################################################################

print('''\n
###############################################################################
#####  Step 1: Preparation of Data Set
###############################################################################      
''')


###############################################################################
#####  Item 1.1 Source citation for your data set
###############################################################################

print('''
###############################################################################
#####  Step 1.1 Source citation for your data set
###############################################################################      
\n''')

# Get Snohomish County assessor data from ftp site
# ftp://ftp.snoco.org/assessor/property_sales/Snohomish%20County%20Sales/
ftp = "ftp://ftp.snoco.org/assessor/property_sales/Snohomish%20County%20Sales/2019.10.08_Entire%20County.xlsx"

print(ftp)

###############################################################################
#####  Item 1.2 Data read from an easily and freely accessible source
###############################################################################

print('''
###############################################################################
#####  Step 1.2 Data read from an easily and freely accessible source
#####  This may take some time to read depending on your internet speed.
###############################################################################      
\n''')

# This may take some time to read depending on your internet speed.
data_excel = pd.read_excel(ftp)

print("\nSave Excel file as a csv file, JosephSimpson-MS3-Data.csv.\n")
# Save excel file as csv
data_excel.to_csv('JosephSimpson-MS3-Data.csv', encoding='utf-8')

print("\n Create a Pandas dataframe.\n")
# Read in data csv file
c1 = pd.read_csv('JosephSimpson-MS3-Data.csv')
df_c1 = pd.DataFrame(c1)

###############################################################################
#####  Item 1.3 Number of observations and attributes
###############################################################################

print('''
###############################################################################
#####  Step 1.3 Number of observations and attributes
###############################################################################      
\n''')

print(df_c1.shape)
# (88623, 54)
print('''\n
     The original data set contains 88623 observation and
     54 attributes.
\n''')

###############################################################################
#####  Item 1.4 Original Data types
###############################################################################

print('''
###############################################################################
#####  Step 1.4 Original Data types
###############################################################################      
\n''')

print(df_c1.info())

print('''\n
      RangeIndex: 88623 entries, 0 to 88622
Data columns (total 54 columns):
Unnamed: 0              88623 non-null int64
LRSN                    88623 non-null int64
Parcel_Id               88623 non-null int64
Status                  88623 non-null object
SD_Nbr                  88623 non-null int64
Nbhd                    88623 non-null int64
TRSQ                    88619 non-null float64
Prop_Class              88623 non-null int64
PropertyStreet          88612 non-null object
OwnerName1              88589 non-null object
Sale_Date               88623 non-null object
Sale_Price              88623 non-null int64
Excise_Nbr              88618 non-null object
Deed_Type               88484 non-null object
Sale_Qual_Code          88623 non-null object
V/I                     88623 non-null object
LL1_Type                85481 non-null object
LL1_Calc_Method         85481 non-null float64
LL1_Acres               85481 non-null float64
Ll1_SqFt                85481 non-null float64
Ll1_FF                  85481 non-null float64
Ll1_EFF                 85481 non-null float64
Total_Land_Size         88623 non-null float64
zoning                  67622 non-null object
Ll1_Inf_1               34169 non-null object
Last_Value_Update       88623 non-null object
Imp_Value               88623 non-null int64
Mkt_Land_Value          88623 non-null int64
Total_Market_Value      88623 non-null int64
Land_Use_Value          88623 non-null int64
Change_Reason           88623 non-null int64
eff_year                88623 non-null int64
Exten                   82469 non-null object
Imp_Type                82469 non-null object
Impr_Width              82469 non-null float64
Imp_Length              82469 non-null float64
Grade                   82469 non-null float64
Grade_Desc              82469 non-null object
Yr_Blt                  82469 non-null float64
Eff_Yr_Blt              82469 non-null float64
House_Type              82469 non-null float64
Hse_Type_Desc           82457 non-null object
Bedrooms                82469 non-null float64
MH_Length               82469 non-null float64
MH_Width                82469 non-null float64
Imp_Size                82469 non-null float64
B_L_SqFt                77986 non-null float64
1st_SqFt                77986 non-null float64
Upper_SqFt              77986 non-null float64
Total_SqFt              77986 non-null float64
PCT_Comp                82469 non-null float64
mkt_rdf                 82469 non-null float64
Transfer_Update_Date    88623 non-null object
Cert_Value_Ratio        88623 non-null float64
dtypes: float64(24), int64(13), object(17)
\n''')
    
###############################################################################
#####  Item 1.5  Distribution of numerical variables
###############################################################################

print('''
###############################################################################
#####  Step 1.5  Distribution of numerical variables
#####            24 float data types -- 13 integer data types
#####            Some of these numeric values may be changed to categories
###############################################################################      
\n''')
    
df_c1['LRSN'].describe()

print('''\n
df_c1['LRSN'].describe()
 
count    8.862300e+04
mean     2.348755e+06
std      1.675202e+06
min      3.000180e+05
25%      1.094356e+06
50%      1.204168e+06
75%      4.374898e+06
max      4.431266e+06
Name: LRSN, dtype: float64      
\n''')

df_c1['Parcel_Id'].describe()

print('''\n
df_c1['Parcel_Id'].describe()
 
count    8.862300e+04
mean     4.475321e+12
std      9.502216e+12
min      3.705000e+11
25%      6.032500e+11
50%      9.020000e+11
75%      1.143400e+12
max      3.210080e+13
Name: Parcel_Id, dtype: float64      
\n''')

df_c1['SD_Nbr'].describe()

print('''\n
df_c1['SD_Nbr'].describe()
 
count    88623.000000
mean        96.301197
std        143.829380
min          1.000000
25%          4.000000
50%         15.000000
75%        201.000000
max        417.000000
Name: SD_Nbr, dtype: float64      
\n''')

df_c1['Nbhd'].describe()

print('''\n
df_c1['Nbhd'].describe()

count    8.862300e+04
mean     3.176795e+06
std      1.370209e+06
min      1.101006e+06
25%      2.207000e+06
50%      3.304001e+06
75%      4.304000e+06
max      6.101000e+06
Name: Nbhd, dtype: float64      
\n''')

df_c1['TRSQ'].describe()

print('''\n
df_c1['TRSQ'].describe()
 
count    8.861900e+04
mean     2.846109e+06
std      1.364846e+06
min      5.000000e+00
25%      2.705093e+06
50%      2.805082e+06
75%      2.906062e+06
max      2.828053e+08
Name: TRSQ, dtype: float64      
\n''')

df_c1['Prop_Class'].describe()

print('''\n
df_c1['Prop_Class'].describe()
 
count    88623.000000
mean       151.841813
std        155.045732
min          1.000000
25%        111.000000
50%        111.000000
75%        122.000000
max        941.000000
Name: Prop_Class, dtype: float64      
\n''')

df_c1['Sale_Price'].describe()

print('''\n
df_c1['Sale_Price'].describe()

count    8.862300e+04
mean     8.808003e+05
std      3.942812e+06
min      1.000000e+00
25%      2.900000e+05
50%      4.030000e+05
75%      5.599500e+05
max      1.455000e+08
Name: Sale_Price, dtype: float64      
\n''')

df_c1['LL1_Calc_Method'].describe()

print('''\n
df_c1['LL1_Calc_Method'].describe()
 
count    85481.000000
mean        11.536283
std          8.855507
min          2.000000
25%          7.000000
50%          7.000000
75%         14.000000
max         46.000000
Name: LL1_Calc_Method, dtype: float64     
\n''')

df_c1['LL1_Acres'].describe()

print('''\n
df_c1['LL1_Acres'].describe()

count    85481.000000
mean         1.133637
std        114.959965
min          0.000000
25%          0.090000
50%          0.170000
75%          0.275500
max      32200.000000
Name: LL1_Acres, dtype: float64      
\n''')

df_c1['Ll1_SqFt'].describe()

print('''\n
df_c1['Ll1_SqFt'].describe()
 
count    8.548100e+04
mean     9.170013e+03
std      3.217126e+04
min      0.000000e+00
25%      2.613600e+03
50%      6.000000e+03
75%      9.500000e+03
max      3.377207e+06
Name: Ll1_SqFt, dtype: float64      
\n''')

df_c1['Ll1_FF'].describe()

print('''\n
df_c1['Ll1_FF'].describe()
Out[157]: 
count     85481.000000
mean          3.462897
std         528.202056
min           0.000000
25%           0.000000
50%           0.000000
75%           0.000000
max      109000.000000
Name: Ll1_FF, dtype: float64      
\n''')

df_c1['Ll1_EFF'].describe()

print('''\n
df_c1['Ll1_EFF'].describe()
 
count    85481.000000
mean         0.617447
std         21.627587
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max       6000.000000
Name: Ll1_EFF, dtype: float64      
\n''')

df_c1['Total_Land_Size'].describe()

print('''\n)
df_c1['Total_Land_Size'].describe()
 
count    88623.000000
mean         0.736640
std         24.620292
min          0.000000
25%          0.071100
50%          0.170000
75%          0.290000
max       7283.000000
Name: Total_Land_Size, dtype: float64
\n''')

df_c1['Imp_Value'].describe()

print('''\n
df_c1['Imp_Value'].describe()
 
count    8.862300e+04
mean     3.267249e+05
std      1.435167e+06
min      0.000000e+00
25%      1.559000e+05
50%      2.234000e+05
75%      3.100000e+05
max      1.074198e+08
Name: Imp_Value, dtype: float64      
\n''')

df_c1['Mkt_Land_Value'].describe()

print('''\n)
df_c1['Mkt_Land_Value'].describe()
Out[161]: 
count    8.862300e+04
mean     2.453325e+05
std      4.611706e+05
min      0.000000e+00
25%      1.450000e+05
50%      1.950000e+05
75%      2.810000e+05
max      2.578010e+07
Name: Mkt_Land_Value, dtype: float64
\n''')

df_c1['Total_Market_Value'].describe()

print('''\n
df_c1['Total_Market_Value'].describe()
 
count    8.862300e+04
mean     5.720573e+05
std      1.722510e+06
min      0.000000e+00
25%      3.203000e+05
50%      4.284000e+05
75%      5.721000e+05
max      1.228684e+08
Name: Total_Market_Value, dtype: float64      
\n''')

df_c1['Land_Use_Value'].describe()

print('''\n
df_c1['Land_Use_Value'].describe()
 
count     88623.000000
mean        131.066427
std        5903.659230
min           0.000000
25%           0.000000
50%           0.000000
75%           0.000000
max      828500.000000
Name: Land_Use_Value, dtype: float64      
\n''')

df_c1['Change_Reason'].describe()

print('''\n
df_c1['Change_Reason'].describe()
 
count    88623.000000
mean         1.267955
std          1.223772
min          1.000000
25%          1.000000
50%          1.000000
75%          1.000000
max         41.000000
Name: Change_Reason, dtype: float64      
\n''')

df_c1['eff_year'].describe()

print('''\n
df_c1['eff_year'].describe()
 
count    8.862300e+04
mean     2.018990e+07
std      2.236719e+03
min      2.007010e+07
25%      2.019010e+07
50%      2.019010e+07
75%      2.019010e+07
max      2.020010e+07
Name: eff_year, dtype: float64      
\n''')

df_c1['Impr_Width'].describe()

print('''\n
df_c1['Impr_Width'].describe()
 
count    82469.000000
mean         1.318023
std         15.010976
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max       2016.000000
Name: Impr_Width, dtype: float64      
\n''')

df_c1['Imp_Length'].describe()

print('''\n
df_c1['Imp_Length'].describe()
 
count    82469.000000
mean         2.989535
std         18.681302
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max       2016.000000
Name: Imp_Length, dtype: float64      
\n''')

df_c1['Grade'].describe()

print('''\n
df_c1['Grade'].describe()
 
count    82469.000000
mean        45.472299
std          6.350645
min         15.000000
25%         45.000000
50%         45.000000
75%         49.000000
max         75.000000
Name: Grade, dtype: float64      
\n''')

df_c1['Yr_Blt'].describe()

print('''\n
df_c1['Yr_Blt'].describe()
 
count    82469.000000
mean      1990.341389
std         26.879327
min          0.000000
25%       1978.000000
50%       1997.000000
75%       2010.000000
max       2020.000000
Name: Yr_Blt, dtype: float64      
\n''')

df_c1['Eff_Yr_Blt'].describe()

print('''\n
 df_c1['Eff_Yr_Blt'].describe()
Out[171]: 
count    82469.000000
mean      1991.082831
std         26.317365
min          0.000000
25%       1979.000000
50%       1997.000000
75%       2010.000000
max       2020.000000
Name: Eff_Yr_Blt, dtype: float64      
\n''')
    
df_c1['House_Type'].describe()    

print('''\n
df_c1['House_Type'].describe()    
 
count    82469.000000
mean        18.721180
std         13.245768
min          0.000000
25%         11.000000
50%         17.000000
75%         17.000000
max         96.000000
Name: House_Type, dtype: float64      
\n''')

df_c1['Bedrooms'].describe()

print('''\n
df_c1['Bedrooms'].describe()
 
count    82469.000000
mean         3.587457
std        108.328996
min          0.000000
25%          3.000000
50%          3.000000
75%          4.000000
max      31111.000000
Name: Bedrooms, dtype: float64      
\n''')

df_c1['MH_Length'].describe()

print('''\n
df_c1['MH_Length'].describe()
 
count    82469.000000
mean         2.853618
std         12.282280
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max         76.000000
Name: MH_Length, dtype: float64      
\n''')

df_c1['MH_Width'].describe()

print('''\n
df_c1['MH_Width'].describe()
 
count    82469.000000
mean         1.192921
std          5.272877
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max         56.000000
Name: MH_Width, dtype: float64      
\n''')

df_c1['Imp_Size'].describe()

print('''\n
df_c1['Imp_Size'].describe()
 
count     82469.000000
mean         69.701342
std         832.268321
min           0.000000
25%           0.000000
50%           0.000000
75%           0.000000
max      223323.000000
Name: Imp_Size, dtype: float64      
\n''')

df_c1['B_L_SqFt'].describe()

print('''\n
df_c1['B_L_SqFt'].describe()
 
count    77986.000000
mean       187.524684
std        389.587937
min          0.000000
25%          0.000000
50%          0.000000
75%         40.000000
max       5768.000000
Name: B_L_SqFt, dtype: float64      
\n''')

df_c1['1st_SqFt'].describe()

print('''\n
df_c1['1st_SqFt'].describe()
 
count    77986.000000
mean      1192.716706
std        463.032286
min         32.000000
25%        886.000000
50%       1124.000000
75%       1403.000000
max      22788.000000
Name: 1st_SqFt, dtype: float64      
\n''')

df_c1['Upper_SqFt'].describe()

print('''\n
df_c1['Upper_SqFt'].describe()
 
count    77986.000000
mean       618.899597
std        652.613347
min          0.000000
25%          0.000000
50%        540.000000
75%       1173.000000
max       5992.000000
Name: Upper_SqFt, dtype: float64      
\n''')

df_c1['Total_SqFt'].describe()

print('''\n
df_c1['Total_SqFt'].describe()
 
count    77986.000000
mean      1999.140987
std        832.719091
min         32.000000
25%       1394.000000
50%       1893.000000
75%       2487.000000
max      22788.000000
Name: Total_SqFt, dtype: float64      
\n''')

df_c1['PCT_Comp'].describe()

print('''\n
df_c1['PCT_Comp'].describe()
 
count    82469.000000
mean        99.510786
std          5.055980
min          0.000000
25%        100.000000
50%        100.000000
75%        100.000000
max        100.000000
Name: PCT_Comp, dtype: float64      
\n''')

df_c1['mkt_rdf'].describe()

print('''\n
df_c1['mkt_rdf'].describe()
 
count    82469.000000
mean       100.057197
std          6.595222
min          0.000000
25%        100.000000
50%        100.000000
75%        100.000000
max        300.000000
Name: mkt_rdf, dtype: float64      
\n''')

df_c1['Cert_Value_Ratio'].describe()

print('''\n
df_c1['Cert_Value_Ratio'].describe()
 
count     88623.000000
mean          6.653086
std        1206.201033
min           0.000000
25%           0.939000
50%           1.062900
75%           1.238100
max      346600.000000
Name: Cert_Value_Ratio, dtype: float64      
\n''')

###############################################################################
#####  Item 1.6  Distribution of categorical variables
###############################################################################

print('''
###############################################################################
#####  Step 1.6  Distribution of categorical variables
#####            17 object types that may be categories  
#####            
###############################################################################      
\n''')

df_c1['Status'].describe()
df_c1['Status'].value_counts()
print('''\n 
df_c1['Status'].describe()
 
count     88623
unique        2
top           A
freq      87453
Name: Status, dtype: object      

df_c1['Status'].value_counts()
 
A    87453
T     1170
Name: Status, dtype: int64
\n''')

df_c1['PropertyStreet'].describe()
df_c1['PropertyStreet'].value_counts()

print('''\n
df_c1['PropertyStreet'].describe()
 
count                         88612
unique                        72244
top       UNKNOWN UNKNOWN,UNKNOWN,,
freq                           1749
Name: PropertyStreet, dtype: object     

df_c1['PropertyStreet'].value_counts()
Out[190]: 
UNKNOWN UNKNOWN,UNKNOWN,,                   1749
UNKNOWN,UNKNOWN,WA,USA                       295
UNKNOWN,WA,USA                               250
UNKNOWN, UNKNOWN,UNKNOWN,WA,                 203
UNKNOWN,UNKNOWN,WA,                          135
4124 COLBY AVE,EVERETT,WA,98203-2306,USA      90
8330 276TH PL NW UNIT 7,STANWOOD,WA,9829      88
1529 63RD ST SE UNIT D-1,EVERETT,WA,9820      88
2709 LINCOLN WAY,LYNNWOOD,WA,98087-5622,      73
19802 MERIDIAN PL W,BOTHELL,WA,98012,         38
3728 SUNNYSIDE BLVD,MARYSVILLE,WA,98270-      36
19307 35TH AVE SE,BOTHELL,WA,98012,USA        36
2121 148TH ST SW,LYNNWOOD,WA,98087-5901,      33
1605 150TH ST SW,LYNNWOOD,WA,98087-8713,      32
5810 FLEMING ST,EVERETT,WA,98203,USA          32
14TH ST,EVERETT,WA,98201,USA                  32
9523 55TH AVE NE,MARYSVILLE,WA,98270-244      30
212 OLD OWEN RD,SULTAN,WA,98294,USA           28
13411 ASH WAY,EVERETT,WA,98204-6328,USA       26
16021 MANOR WAY,LYNNWOOD,WA,98087,            26
7104 265TH ST NW,STANWOOD,WA,98292-6250,      21
21326 45TH AVE SE,BOTHELL,WA,98021-7919,      21
7907 230TH ST SW,EDMONDS,WA,98026-8731,U      19
15033 18TH AVE W,LYNNWOOD,WA,98087-8703,      18
19330 WINESAP RD,BOTHELL,WA,98012,USA         18
22529 39TH AVE SE,BOTHELL,WA,98021-7942,      15
7325 RAINIER DR,EVERETT,WA,98203,             15
UNKNOWN,,UNKNOWN,WA,                          14
164TH ST SW,LYNNWOOD,WA,98087,                13
105 COX ST,ARLINGTON,WA,98223,USA             13

2906 103RD AVE SE,LAKE STEVENS,WA,98258,       1
121 N BOGART AVE,GRANITE FALLS,WA,98252-       1
12916 54TH AVE SE,EVERETT,WA,98208-9530,       1
4232 113TH AVE SE,SNOHOMISH,WA,98290-557       1
616 5TH AVE S,EDMONDS,WA,98020-3404,USA        1
8635 NORDIC WAY,STANWOOD,WA,98292,             1
7730 196TH ST SW UNIT 5,EDMONDS,WA,98026       1
15415 35TH AVE W UNIT F-103,LYNNWOOD,WA,       1
11325 19TH AVE SE UNIT 312D,EVERETT,WA,9       1
215 100TH ST SW UNIT B305,EVERETT,WA,982       1
18227 WOODLANDS WAY,ARLINGTON,WA,98223-5       1
17821 BROOK BLVD,BOTHELL,WA,98012-6446,U       1
5818 WETMORE AVE SE,EVERETT,WA,98203,USA       1
19329 6TH DR SE,BOTHELL,WA,98012,              1
1122 135TH ST SW,EVERETT,WA,98204-7313,        1
6415 278TH ST NW,STANWOOD,WA,98292,            1
12410 54TH DR NE,MARYSVILLE,WA,98271,          1
2607 ROCKEFELLER AVE,EVERETT,WA,98201-29       1
12517 CHAIN LAKE RD,SNOHOMISH,WA,98290-3       1
14216 STATE ROUTE 530 NE,ARLINGTON,WA,98       1
10919 ALGONQUIN RD,WOODWAY,WA,98020-6108       1
13160 164TH AVE SE,MONROE,WA,98272,            1
3611 174TH PL NE,ARLINGTON,WA,98223-6338       1
844 OLYMPIC BLVD,EVERETT,WA,98203-1815,U       1
2222 120TH PL SE,EVERETT,WA,98208-6227,U       1
13407 35TH AVE SE,MILL CREEK,WA,98012-89       1
10104 9TH AVE W,EVERETT,WA,98204-3776,         1
11704 TULARE WAY W,MARYSVILLE,WA,98271,U       1
17605 42ND AVE SE,BOTHELL,WA,98012-7857,       1
12015 MARINE DR LOT 1643,MARYSVILLE,WA,9       1
Name: PropertyStreet, Length: 72244, dtype: int64
\n''')

df_c1['OwnerName1'].describe()
df_c1['OwnerName1'].value_counts() 

print('''\n
df_c1['OwnerName1'].describe()
 
count        88589
unique       66850
top       SSHI LLC
freq           407
Name: OwnerName1, dtype: object      

df_c1['OwnerName1'].value_counts() 
 
SSHI LLC                                407
PACIFIC RIDGE-DRH LLC                   349
LENNAR NORTHWEST INC                    321
HIGHLAND REALTY ADVISORS LLC            269
PULTE HOMES OF WASHINGTON INC           184
CR PARK 120 COMMUNITIES LLC             152
CORNERSTONE HOMES NW LLC                151
PRH LLC                                 141
PACIFIC RIDGE - DRH LLC                 120
MOUNTLAKE TERRACE INVESTORS             112
RMH LLC                                 103
SSHI LLC DBA DR HORTON                  100
QUADRANT CORPORATION                     99
FORESTAR (USA) REAL ESTATE GROUP INC     94
DUONG RANG V & DINH LE-THAO T            91
RORDAME XORIA & GAUR RAJESH              88
CARPENTER MARK E                         88
LANDSVERK QUALITY HOMES INC              79
CREEKSTONE DEVELOPMENT LLC               78
RM HOMES LLC                             76
HARBOUR HOMES LLC                        76
ACME HOMES LLC                           74
MAINVUE WA  LLC                          74
GOLDEN QIFANG HOLDINGS LP                73
IH6 PROPERTY WASHINGTON LP               68
D R HORTON                               67
WESTVIEW RIDGE HOMES LLC                 66
EVARONE JACK W & MARLENE I               64
LGI HOMES-WASHINGTON LLC                 62
SELECT HOMES INC                         62

BARRETT BRIAN                             1
HEPWORTH KEVIN                            1
ABRAHAM RONNY N & VARKEY BONNY S          1
ODISHO WALTER & CAROLYN Y                 1
NEBENFUHR JUNG HEE/KIM MIN SUNG           1
FARMER RICK J & SIONY                     1
KAMBHAMPATI SATYA S & BHAMIDIPATI MA      1
FOX GARY/FOX NICOLE                       1
GALLO GWENDOLYN                           1
KEMP PATRICK & WALLER MICHELE             1
BUI JENNY & PHAM PETER                    1
HENCZ BRANDON & DANAE                     1
CHARAIS RICHARD L & LORETTA STARR         1
BERGAMO AARON H & BRANDI L                1
CHINN SHERENE A/GO BETSY Y                1
CORTES CESAR BAILON                       1
RUCHIRAT PETER                            1
HARRIS AMANDA                             1
SARKIS SAMIR A                            1
COBBS RANDALL & HEATHER                   1
PATTERSON DIANA C                         1
WEISHAAR JEANNE MARIE                     1
MORGAN JODELLE                            1
MEDHANE ELSA/EZE OKEZIE                   1
LARSEN THOMAS W & MOLLY A                 1
AL ZEBEN REEMA/BULMAN PATRICK             1
DAVIS WILLIAM P SR                        1
KAMINISKI MARK & RUTH A                   1
VASILCHENKO NAZARY & LESYA                1
GUILLEN GILBERT & SANDRA                  1
Name: OwnerName1, Length: 66850, dtype: int64

\n''')   

df_c1['Sale_Date'].describe()
df_c1['Sale_Date'].value_counts() 

print('''\n
df_c1['Sale_Date'].describe()
Out[193]: 
count          88623
unique          1675
top       2019-06-06
freq             188
Name: Sale_Date, dtype: object      

df_c1['Sale_Date'].describe()

count          88623
unique          1675
top       2019-06-06
freq             188
Name: Sale_Date, dtype: object

df_c1['Sale_Date'].value_counts() 
Out[194]: 
2019-06-06    188
2017-07-07    187
2015-04-22    182
2016-04-25    169
2019-08-27    165
2018-06-05    165
2017-10-30    164
2019-02-21    155
2016-08-02    155
2019-08-07    154
2015-12-10    145
2019-05-06    143
2017-08-01    142
2017-07-05    142
2016-07-20    140
2017-09-01    138
2015-03-04    134
2018-09-12    133
2018-06-28    133
2017-07-24    132
2017-07-11    132
2017-06-05    131
2018-05-01    129
2019-01-02    129
2017-07-26    128
2015-06-24    128
2015-08-19    127
2017-06-20    125
2018-09-10    125
2017-06-26    125

2018-11-17      1
2018-01-21      1
2016-06-18      1
2017-12-10      1
2017-04-02      1
2019-04-21      1
2017-01-22      1
2017-02-12      1
2015-11-08      1
2018-01-15      1
2018-06-17      1
2019-09-21      1
2017-03-26      1
2015-02-15      1
2015-01-17      1
2018-05-27      1
2019-01-05      1
2019-08-31      1
2016-12-11      1
2018-12-08      1
2015-06-07      1
2018-11-25      1
2015-08-30      1
2017-08-27      1
2017-01-21      1
2017-01-07      1
2015-02-07      1
2015-01-11      1
2019-02-17      1
2019-09-29      1
Name: Sale_Date, Length: 1675, dtype: int64

\n''')

df_c1['Excise_Nbr'].describe()
df_c1['Excise_Nbr'].value_counts()    
    
print('''\n
df_c1['Excise_Nbr'].describe()
 
count       88618
unique      82155
top       1196150
freq          112
Name: Excise_Nbr, dtype: object   

df_c1['Excise_Nbr'].value_counts()
 
1196150    112
1141733     95
E126848     88
1101251     88
E125349     88
E047471     82
E106386     76
1091827     76
1188007     76
E104950     76
1110409     73
1093474     70
E115278     60
E114953     59
E061741     46
E110291     45
1185136     45
E095882     44
E103453     42
1165375     42
E061704     40
E120130     40
E111207     40
1140939     37
1136677     37
1155569     36
1104591     36
E076212     36
1198215     36
E109910     35

E081345      1
E105324      1
E126586      1
1160940      1
E112400      1
E123571      1
E057778      1
E105652      1
E092034      1
E119987      1
E101244      1
1113369      1
E089786      1
E051427      1
E125219      1
E110881      1
E103898      1
E120850      1
E071069      1
E048873      1
E112568      1
1128969      1
E078755      1
E108190      1
E115951      1
E051401      1
1185493      1
1085402      1
E107260      1
E126236      1
Name: Excise_Nbr, Length: 82155, dtype: int64
   
\n''')   
    
df_c1['Deed_Type'].describe()
df_c1['Deed_Type'].value_counts()     
    
print('''\n
df_c1['Deed_Type'].describe()
 
count     88484
unique       12
top           W
freq      80303
Name: Deed_Type, dtype: object   

df_c1['Deed_Type'].value_counts()
 
W     80303
X      3591
BS     2601
WP     1836
R        82
w        59
Q         4
QC        4
W1        1
S         1
bs        1
WQ        1
Name: Deed_Type, dtype: int64         
\n''')  
    
df_c1['Sale_Qual_Code'].describe()
df_c1['Sale_Qual_Code'].value_counts()    
    
print('''\n
df_c1['Sale_Qual_Code'].describe()
 
count     88623
unique        7
top           Q
freq      74800
Name: Sale_Qual_Code, dtype: object      

df_c1['Sale_Qual_Code'].value_counts()  
 
Q     74800
M      4606
ZM     3698
E      3106
Z      1577
V       745
B        91
Name: Sale_Qual_Code, dtype: int64
\n''')   
    
df_c1['V/I'].describe()
df_c1['V/I'].value_counts()        
    
print('''\n
df_c1['V/I'].describe()
 
count     88623
unique       95
top        VVVV
freq      65747
Name: V/I, dtype: object    

df_c1['V/I'].value_counts()
 
VVVV            65747
AA               8090
I                7286
9999             3840
V                2486
XX                889
T                  21
04                  3
appraisers          3
M73                 3
05                  3
40                  3
appealdedtyp        3
104                 3
11                  3
0                   3
AGR3                3
AGR4                3
09                  3
01                  3
M8NA                3
102                 3
futurechgr          3
H64                 3
OSG1                3
21                  3
soilid              3
31                  3
2                   3
08                  3
 
H44                 3
A71                 3
20                  3
D12                 3
D31                 3
AGR1                3
IN                  3
D43                 3
D33                 3
AGR2                3
07                  3
105                 3
AGR5                3
06                  3
103                 3
101                 3
1                   3
H32                 3
AR                  3
36                  3
incunittype         3
A74                 3
10                  3
OT                  3
D32                 3
N8NA                3
4                   3
D22                 3
A73                 3
H53                 3
Name: V/I, Length: 95, dtype: int64
  
\n''')    
    
df_c1['LL1_Type'].describe()
df_c1['LL1_Type'].value_counts()            
    
print('''\n
df_c1['LL1_Type'].describe()
 
count     85481
unique      116
top          A3
freq      22981
Name: LL1_Type, dtype: object      

df_c1['LL1_Type'].value_counts()
Out[205]: 
A3    22981
A2     8480
A4     7664
B2     5444
C2     4117
G4     3472
B4     3262
95     2997
A6     2916
94     2724
93     1955
B6     1753
A1     1699
C6     1355
73      983
F1      912
C4      811
77      798
A7      781
92      676
B1      595
91      585
A9      546
A5      541
90      504
G6      474
C3      411
88      394
B7      344
76      323
 
U8       11
O4        9
MN        9
G1        8
L8        7
U7        7
GC        5
56        5
B9        5
DV        5
28        5
46        3
14        2
W5        2
62        2
26        2
LF        2
66        2
R3        2
U9        2
O1        2
G9        1
54        1
B8        1
41        1
32        1
4         1
UW        1
V6        1
U4        1
Name: LL1_Type, Length: 116, dtype: int64
\n''')    
    
df_c1['zoning'].describe()
df_c1['zoning'].value_counts()         
    
print('''\n
df_c1['zoning'].describe()
 
count       67622
unique        304
top       SNC R-5
freq         8940
Name: zoning, dtype: object   

df_c1['zoning'].value_counts()
 
SNC R-5                  8940
SNC R-9,600              3778
SNC R-7,200              2613
MAR R4.5 SFM             2436
MAR R6.5 SFH             2371
SNC MR                   2256
SNC PRD-9,600            2246
EVE R-2                  2224
EVE R-1                  2118
SNC R-8,400              1554
LYN RS8                  1523
SNC LDMR                 1445
EVE R-3                  1441
EDM RS-8                 1405
ARL RLMD                 1103
MOU RS 7200               985
LAK R-7,200               796
SNC UC                    784
MIL PRD 7200              715
LAK SR                    691
LAK UR                    685
MIL LDR                   650
SNO SFRES                 647
EDM RM-1.5                628
LAK R-9,600               600
MON UR9600                596
BOT R 9,600               585
SNC F                     548
ARL RMD                   544
LYN RMM                   540

MUK BP                      2
STA LI                      2
LAK PBD                     2
SNC RFS                     2
BOT R 9,600, NCFWCHPA       2
DAR LI                      2
MOU MHP                     2
LYN PRC                     2
EVE SPD                     2
EDM MU                      2
SNC MHP                     2
LAK Interim MHP             2
BRI BN                      2
SNC PIP                     2
SNC RI                      2
LYN B4                      1
SNC FS                      1
EDM OR                      1
ARL MS                      1
MON LOS                     1
LAK P/SP                    1
EVE WRM                     1
MON PS                      1
MAR BP                      1
SNC TRIBES                  1
SNC PRUD                    1
MUK RD9.6(S)                1
LAK LB                      1
MOU SDD C/R                 1
SNO AI                      1
Name: zoning, Length: 304, dtype: int64
\n''')  

df_c1['Ll1_Inf_1'].describe()
df_c1['Ll1_Inf_1'].value_counts()  

print('''\n
df_c1['Ll1_Inf_1'].describe()
 
count     34169
unique      213
top          e1
freq       3383
Name: Ll1_Inf_1, dtype: object    

df_c1['Ll1_Inf_1'].value_counts() 

e1    3383
b1    3143
e4    2984
a1    2846
b4    2710
f1    1273
b6    1148
c1    1100
A1    1026
t1     846
B4     717
p1     533
e8     512
f4     490
B6     477
p4     463
A4     417
d2     379
t4     371
d1     367
a4     353
c4     337
T1     336
U9     333
54     330
v1     264
B1     253
D2     245
T4     244
E4     211

n7       2
O8       2
n8       2
T9       2
F0       2
g9       1
z8       1
p0       1
p        1
G1       1
M6       1
f0       1
i8       1
G7       1
d7       1
41       1
G3       1
b9       1
x0       1
5        1
p9       1
14       1
3        1
t9       1
R8       1
A2       1
o0       1
x        1
R9       1
O0       1
Name: Ll1_Inf_1, Length: 213, dtype: int64
\n''')

df_c1['Last_Value_Update'].describe()
df_c1['Last_Value_Update'].value_counts()  

print('''\n
df_c1['Last_Value_Update'].describe()
 
count          88623
unique           225
top       2019-05-30
freq           21465
Name: Last_Value_Update, dtype: object 

df_c1['Last_Value_Update'].value_counts() 
 
2019-05-30    21465
2019-05-23    18677
2019-05-24    17780
2019-05-25     9678
2019-05-21     9301
2019-05-29     4538
2019-08-29      304
2019-08-05      261
2019-09-03      246
2019-07-22      210
2019-08-20      199
2019-08-21      188
2019-08-15      182
2019-06-27      181
2019-07-18      177
2019-08-26      169
2019-06-13      165
2019-08-28      153
2019-08-14      150
2019-06-26      146
2019-07-03      142
2019-08-13      140
2019-08-27      138
2019-08-19      115
2019-06-07      113
2019-06-25      112
2019-06-20      111
2019-07-16      110
2019-06-17      110
2019-08-07      105
 
2016-05-26        1
2014-05-22        1
2015-06-08        1
2017-10-16        1
2017-09-25        1
2016-10-19        1
2016-12-15        1
2015-08-17        1
2019-06-22        1
2019-09-17        1
2018-07-12        1
2018-06-21        1
2017-07-20        1
2017-07-17        1
2017-06-12        1
2017-09-19        1
2017-08-26        1
2016-03-24        1
2015-11-17        1
2018-06-18        1
2017-02-22        1
2018-07-26        1
2018-03-13        1
2015-08-18        1
2016-03-17        1
2017-02-07        1
2018-04-04        1
2018-09-04        1
2018-08-28        1
2017-03-23        1
Name: Last_Value_Update, Length: 225, dtype: int64     
\n''')

df_c1['Exten'].describe()
df_c1['Exten'].value_counts()  

print('''\n
df_c1['Exten'].describe()
 
count     82469
unique       81
top         R01
freq      75644
Name: Exten, dtype: object      

df_c1['Exten'].value_counts()
 
R01    75644
R02     5509
R03      622
R04      153
R05       70
R06       46
R07       36
R08       34
R09       34
R10       31
R11       27
R12       24
R13       22
R14       19
R15       15
R17       14
R16       12
R19        9
R18        8
R20        7
R21        6
R27        5
R23        5
R24        5
R31        4
R22        4
R34        3
R33        3
R25        3
R35        3
 
R47        2
R67        2
R57        2
R50        2
R73        2
R62        2
R52        2
R44        2
R54        2
R63        2
R66        2
R61        2
R72        2
R45        2
R60        2
R41        1
R77        1
R40        1
R42        1
R43        1
R80        1
R38        1
R74        1
R39        1
R:0        1
R37        1
R78        1
R75        1
R76        1
R79        1
Name: Exten, Length: 81, dtype: int64
\n''')

df_c1['Imp_Type'].describe()
df_c1['Imp_Type'].value_counts()  
  
print('''\n
df_c1['Imp_Type'].describe()
 
count     82469
unique        2
top       DWELL
freq      78046
Name: Imp_Type, dtype: object      

df_c1['Imp_Type'].value_counts() 
Out[215]: 
DWELL    78046
MHOME     4423
Name: Imp_Type, dtype: int64
\n''')

df_c1['Grade_Desc'].describe()
df_c1['Grade_Desc'].value_counts()      

print('''\n
df_c1['Grade_Desc'].describe()
 
count     82469
unique        9
top         Avg
freq      41145
Name: Grade_Desc, dtype: object  

df_c1['Grade_Desc'].value_counts()
 
Avg        41145
Avg+       15529
Fair        8657
Good        8062
Avg-        6014
V Gd        1593
Low         1263
Exclt        107
Sub Std       99
Name: Grade_Desc, dtype: int64  
\n''')

df_c1['Hse_Type_Desc'].describe()
df_c1['Hse_Type_Desc'].value_counts() 

print('''\
df_c1['Hse_Type_Desc'].describe()
 
count     82457
unique       17
top       2 Sty
freq      32802
Name: Hse_Type_Desc, dtype: object  

df_c1['Hse_Type_Desc'].value_counts() 
 
2 Sty          32802
1 Sty          23347
Split Entry     5864
1 Sty B         4121
2+ Sty          3782
Dbl Wide        3131
2 Sty B         2921
Tri Level       2417
1 1/2 Sty       1724
Sgl Wide        1176
1 1/2 Sty B      998
Trpl Wide         96
2+ Sty B          43
Dbl Wide B        14
Multi Level       11
Quad Level         9
Sgl Wide B         1
Name: Hse_Type_Desc, dtype: int64  
\n''')

df_c1['Transfer_Update_Date'].describe()
df_c1['Transfer_Update_Date'].value_counts()

print('''\n
df_c1['Transfer_Update_Date'].describe()
 
count          88623
unique          1268
top       2019-05-22
freq             617
Name: Transfer_Update_Date, dtype: object      

df_c1['Transfer_Update_Date'].value_counts()
Out[222]: 
2019-05-22    617
2019-09-16    536
2019-09-12    461
2019-10-08    438
2019-09-30    433
2019-10-03    424
2019-09-23    407
2019-01-16    377
2019-10-02    372
2019-08-13    361
2019-09-26    353
2019-09-25    334
2019-08-20    332
2019-10-01    312
2019-09-24    309
2019-09-03    307
2019-05-02    305
2019-10-07    302
2019-06-13    297
2018-10-22    295
2019-07-31    294
2019-09-17    292
2018-10-16    288
2019-08-26    282
2018-01-30    282
2019-09-19    280
2017-03-28    278
2018-10-23    277
2019-08-07    273
2019-06-18    273

2015-09-09      2
2018-09-01      2
2017-12-30      2
2018-05-19      2
2016-05-07      2
2015-01-30      2
2015-05-13      2
2016-11-26      2
2015-03-30      1
2015-02-13      1
2015-07-02      1
2015-01-26      1
2015-01-12      1
2016-11-05      1
2015-03-31      1
2015-12-11      1
2017-05-13      1
2017-12-16      1
2016-12-10      1
2015-05-19      1
2019-03-09      1
2019-02-23      1
2015-03-17      1
2015-03-03      1
2015-02-06      1
2015-05-20      1
2019-05-18      1
2015-01-13      1
2015-11-28      1
2015-06-01      1
Name: Transfer_Update_Date, Length: 1268, dtype: int64
\n''')



###############################################################################
#####  Item 1.7:  A comment on each attribute
###############################################################################

print('''
###############################################################################
#####  Step 1.7: A comment on each attribute
#####    Step 1.7.1 - Numeric attribures - 37 each      
#####    Step 1.7.2 - Catgorical attributes - 17 each       
############################################################################### 

 Step 1.7.1 - Numeric attribures - 37 each        

Unnamed: 0              88623 non-null int64 - index (unused)
LRSN                    88623 non-null int64 - land serial number (may be used)
Parcel_Id               88623 non-null int64 - parcel id (used)
SD_Nbr                  88623 non-null int64 - School District (may be used)
Nbhd                    88623 non-null int64 - Neighborhood (may be used)
TRSQ                    88619 non-null float64 - Location (may be used)
Prop_Class              88623 non-null int64 - Property class (may be used)
Sale_Price              88623 non-null int64 - Sale price (used)
LL1_Calc_Method         85481 non-null float64 - Calculation Method (unused)
LL1_Acres               85481 non-null float64 - Area in acres (used)
Ll1_SqFt                85481 non-null float64 - Area square feet (may be used)
Ll1_FF                  85481 non-null float64 - Front feet (unused)
Ll1_EFF                 85481 non-null float64 - Effctive front feet (unused)
Total_Land_Size         88623 non-null float64 - Total land size (may be used)
Imp_Value               88623 non-null int64 - Improvement value (used)
Mkt_Land_Value          88623 non-null int64 - Land value (used)
Total_Market_Value      88623 non-null int64 - Total market value (used)
Land_Use_Value          88623 non-null int64 - Land use value (unused)
Change_Reason           88623 non-null int64 - Change reason (unused)
eff_year                88623 non-null int64 - Effective year  (unused)
Impr_Width              82469 non-null float64 - Improvement width (unused)
Imp_Length              82469 non-null float64 - Improvement length (unused)
Grade                   82469 non-null float64 - Grade - (may be used)
Yr_Blt                  82469 non-null float64 - Year built (may be used)
Eff_Yr_Blt              82469 non-null float64 - Effctive year built (unused)
House_Type              82469 non-null float64 - House type (used)
Bedrooms                82469 non-null float64 - Number of bedrooms (may be used)
MH_Length               82469 non-null float64 - Moble length (unused)
MH_Width                82469 non-null float64 - Moble width (unused)
Imp_Size                82469 non-null float64 - Improvement size (may be used)
B_L_SqFt                77986 non-null float64 - Basement SF (may be used)
1st_SqFt                77986 non-null float64 - Main floor SF (may be used)
Upper_SqFt              77986 non-null float64 - Upper floor SF (may be used)
Total_SqFt              77986 non-null float64 - Total SF (used)
PCT_Comp                82469 non-null float64 - Percent complete (unused)
mkt_rdf                 82469 non-null float64 - Market? (unused)
Cert_Value_Ratio        88623 non-null float64 - Market ratio (unused)    

Step 1.7.2 - Catgorical attributes - 17 each   

Status                  88623 non-null object - Status of record (used)
PropertyStreet          88612 non-null object - Address  (unused)
OwnerName1              88589 non-null object - Owner (unused)
Sale_Date               88623 non-null object - Sale date (may be used)
Excise_Nbr              88618 non-null object - Tax number (unused)
Deed_Type               88484 non-null object - Deed type (may be used)
Sale_Qual_Code          88623 non-null object - Sale code (unused)
V/I                     88623 non-null object - Condition (unused)
LL1_Type                85481 non-null object - LL1 type (unused)
zoning                  67622 non-null object - Zoning (unused)
Ll1_Inf_1               34169 non-null object - Land factor (unused)
Last_Value_Update       88623 non-null object - Last update (unused)
Exten                   82469 non-null object - Extension (may be used)
Imp_Type                82469 non-null object - Type of improvement (may be used)
Grade_Desc              82469 non-null object - Grade (may be used)
Hse_Type_Desc           82457 non-null object - House type (used)
Transfer_Update_Date    88623 non-null object - Transfer date (unused)
\n''')

###############################################################################
#####   Item 1.8:  Removing cases with missing data and duplicate records
###############################################################################

print('''\n
###############################################################################
#####  Step 1.8:  Removing cases with missing data and duplicate records
###############################################################################
\n''')

print("\nThe original column names are: \n")
print(list(df_c1.columns)) 
print(df_c1.info())

# Next remove nine columns from the original data set that add no value 
# in this analysis task.

print('''\n
      Step 1.8.1 - Remove nine columns from original data set.\n   
''')

# Create new dataframe copy - df_c2
df_c2 = df_c1

# Create modified dataframe - df_c3
df_c3 = df_c2.drop(columns=['PropertyStreet', 'OwnerName1', 
                            'Excise_Nbr','LL1_Calc_Method','zoning','Exten',
                            'Grade_Desc', 'Transfer_Update_Date', 
                            'Cert_Value_Ratio'] )

# Get unique parcel ids. Last time property was sold.
print("\nDrop duplicagte parcel ids. Use only last sale value.\n")
df_c4 = df_c3.drop_duplicates('Parcel_Id')

print("\nGet current dataframe shape. \n")
df_c4.shape
# (75238, 45)

print("\nGet current dataframe information. \n")
print(df_c4.info())

is_unique_id = df_c4['Parcel_Id'].unique()

len(is_unique_id)
# 75238

print("\nDataframe shape and length of unique ids match (75238).\n")
print(len(is_unique_id))
# 75238

# Now get only active records.
is_A = df_c4['Status']=='A'

df_c5 = df_c4[is_A]

# df_c5.shape

print(df_c5.shape)
# (74657, 45)

print("\nNow the Status column may be removed from the dataframe.\n")

df_c5 = df_c4.drop(columns=['Status'] )

print("\nGet dataframe information.\n")
print(df_c5.info())     

print('''
      Int64Index: 75238 entries, 0 to 88622
      Data columns (total 44 columns):
''')

##### Check for vacant property

is_vacant = df_c5['Imp_Value'] == 0
print(sum(is_vacant))  
#  3652


##### Item 1.8.2: Copy modofied file to disk 


print('''\n
       Step 1.8.2: Copy modofied file to disk (Use for local testing)
\n''')

df_c5.to_csv(r'JosephSimpson-MS3-Clean-Data-1.csv')

df_c5_from_disk = pd.read_csv('JosephSimpson-MS3-Data.csv')

# Create new dataframe to support the analysis
df_norm = df_c5.filter(['Nbhd','Sale_Price', 'Total_Land_Size', 'Total_Market_Value', 
                        'Total_SqFt', 'Mkt_Land_Value','Imp_Value',
                        'House_Type'], axis=1)

##### Check df_norm shape
print(df_norm.shape)
##### (75238, 8)

##### Check for nan's in df_norm dataframe
df_norm.loc[:, 'House_Type'].value_counts()

print('''\n
df_norm.loc[:, 'House_Type'].value_counts()
(75238, 7)

17.0    27355
11.0    19809
23.0     5005
12.0     3571
20.0     3139
71.0     2678
18.0     2388
24.0     2166
14.0     1482
74.0      985
15.0      865
77.0       86
21.0       39
72.0       11
26.0        9
27.0        8
96.0        6
0.0         4
Name: House_Type, dtype: int64      
\n''')

##### Check for nan's in df_norm dataframe
print(df_norm.isna().sum())

print('''\n
print(df_norm.isna().sum())
Nbhd                     0
Sale_Price               0
Total_Land_Size          0
Total_Market_Value       0
Total_SqFt            9450
Mkt_Land_Value           0
Imp_Value                0
House_Type            5632
dtype: int64      
\n''')

##### Drop all nan's
# df_norm = df_norm.dropna()
df_norm.fillna(0, inplace=True)
##### Recheck df_norm shape
print(df_norm.shape)
##### (75238, 8)

##### Check for nan's in df_norm dataframe again
print(df_norm.isna().sum())

print('''\n
print(df_norm.isna().sum())
Nbhd                  0
Sale_Price            0
Total_Land_Size       0
Total_Market_Value    0
Total_SqFt            0
Mkt_Land_Value        0
Imp_Value             0
House_Type            0
dtype: int64      
\n''')

###############################################################################
#####  Item 1.9:  Removing outliers 
###############################################################################

print('''\n
###############################################################################
#####  Step 1.9:  Removing outliers 
###############################################################################
\n''')

###############################################################################
##### Input dataframe, remove outliers, return cleaned data in a new dataframe
##### See -- http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
###############################################################################
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & 
                       (df_in[col_name] < fence_high)]
    return df_out    
   

columns = ['Nbhd','Sale_Price', 'Total_Land_Size', 'Total_Market_Value', 
                        'Total_SqFt', 'Mkt_Land_Value', 'Imp_Value']

###############################################################################
##### Remove outliers from dataframe
###############################################################################

def remove_all_outliers(df_out, columns):
    for key in columns:
        df_out = remove_outlier(df_out, key)
        print(df_out.info())       
    return df_out 
   
df_done = remove_all_outliers(df_norm, columns)  

print('''\n
df_done = remove_all_outliers(df_norm, columns)  
<class 'pandas.core.frame.DataFrame'>
Int64Index: 75238 entries, 0 to 88622
Data columns (total 8 columns):
Nbhd                  75238 non-null int64
Sale_Price            75238 non-null int64
Total_Land_Size       75238 non-null float64
Total_Market_Value    75238 non-null int64
Total_SqFt            75238 non-null float64
Mkt_Land_Value        75238 non-null int64
Imp_Value             75238 non-null int64
House_Type            75238 non-null float64
dtypes: float64(3), int64(5)
memory usage: 5.2 MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 70168 entries, 0 to 88614
Data columns (total 8 columns):
Nbhd                  70168 non-null int64
Sale_Price            70168 non-null int64
Total_Land_Size       70168 non-null float64
Total_Market_Value    70168 non-null int64
Total_SqFt            70168 non-null float64
Mkt_Land_Value        70168 non-null int64
Imp_Value             70168 non-null int64
House_Type            70168 non-null float64
dtypes: float64(3), int64(5)
memory usage: 4.8 MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 60866 entries, 0 to 88614
Data columns (total 8 columns):
Nbhd                  60866 non-null int64
Sale_Price            60866 non-null int64
Total_Land_Size       60866 non-null float64
Total_Market_Value    60866 non-null int64
Total_SqFt            60866 non-null float64
Mkt_Land_Value        60866 non-null int64
Imp_Value             60866 non-null int64
House_Type            60866 non-null float64
dtypes: float64(3), int64(5)
memory usage: 4.2 MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 59947 entries, 0 to 88614
Data columns (total 8 columns):
Nbhd                  59947 non-null int64
Sale_Price            59947 non-null int64
Total_Land_Size       59947 non-null float64
Total_Market_Value    59947 non-null int64
Total_SqFt            59947 non-null float64
Mkt_Land_Value        59947 non-null int64
Imp_Value             59947 non-null int64
House_Type            59947 non-null float64
dtypes: float64(3), int64(5)
memory usage: 4.1 MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 59659 entries, 0 to 88614
Data columns (total 8 columns):
Nbhd                  59659 non-null int64
Sale_Price            59659 non-null int64
Total_Land_Size       59659 non-null float64
Total_Market_Value    59659 non-null int64
Total_SqFt            59659 non-null float64
Mkt_Land_Value        59659 non-null int64
Imp_Value             59659 non-null int64
House_Type            59659 non-null float64
dtypes: float64(3), int64(5)
memory usage: 4.1 MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 59312 entries, 0 to 88614
Data columns (total 8 columns):
Nbhd                  59312 non-null int64
Sale_Price            59312 non-null int64
Total_Land_Size       59312 non-null float64
Total_Market_Value    59312 non-null int64
Total_SqFt            59312 non-null float64
Mkt_Land_Value        59312 non-null int64
Imp_Value             59312 non-null int64
House_Type            59312 non-null float64
dtypes: float64(3), int64(5)
memory usage: 4.1 MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 58514 entries, 0 to 88614
Data columns (total 8 columns):
Nbhd                  58514 non-null int64
Sale_Price            58514 non-null int64
Total_Land_Size       58514 non-null float64
Total_Market_Value    58514 non-null int64
Total_SqFt            58514 non-null float64
Mkt_Land_Value        58514 non-null int64
Imp_Value             58514 non-null int64
House_Type            58514 non-null float64
dtypes: float64(3), int64(5)
\n''')


###############################################################################
#####  Item 1.10: Imputing missing values 
###############################################################################

print('''\n
###############################################################################
#####  Step 1.10: Imputing missing values 
###############################################################################
\n''')    

print(df_norm.isna().sum())

print('''\n
print(df_norm.isna().sum())
Nbhd                  0
Sale_Price            0
Total_Land_Size       0
Total_Market_Value    0
Total_SqFt            0
Mkt_Land_Value        0
Imp_Value             0
House_Type            0      
\n''')

print('''\n
      There are no missing values up to this point in the analysis
\n''')

###############################################################################
#####  Item 1.11: Decoding
###############################################################################

print('''\n
###############################################################################
#####  Step 1.11: Decoding 
###############################################################################
\n''')    

##### Check for nan's in df_norm dataframe
df_norm.loc[:, 'House_Type'].value_counts()

print('''\n
df_norm.loc[:, 'House_Type'].value_counts()
Out[261]: 
17.0    27353
11.0    19710
23.0     5005
12.0     3570
20.0     3135
18.0     2387
24.0     2166
14.0     1482
15.0      865
21.0       39
71.0       36
74.0       12
26.0        9
27.0        7
96.0        6
0.0         3
72.0        2
77.0        1
Name: House_Type, dtype: int64      
\n''')

##### Create House_Type_String Column    
print("\nCreate a string copy of the House_Type column.\n")   
print("\nFirst check House_Type unique values.\n") 

df_norm.loc[:,'House_Type'].unique()

print('''\n
     df_norm.loc[:,'House_Type'].unique()
 
array([17., 11., 14., 12., 23., 15., 24., 18., 71., 27., 74., 21., 96.,
       20., 26., 77.,  0., 72.]) 
\n''')

print("\n Add string version of the House_Type column.\n")

df_norm.loc[:,'House_Type_String'] = df_norm['House_Type']    
df_norm.House_Type_String = df_norm.House_Type_String.astype(str)
    
print("\nValidate House_Type_String column.\n")

list(df_norm.columns)

print('''\n
['Nbhd',
 'Sale_Price',
 'Total_Land_Size',
 'Total_Market_Value',
 'Total_SqFt',
 'Mkt_Land_Value',
 'Imp_Value',
 'House_Type',
 'House_Type_String']      
\n''')


print("\nCreate numeric code to string name dictionary.\n")

House_Type_String_Dict = {
        '17.0': 'Standard',
        '11.0': 'Standard',
        '23.0': 'Standard',
        '12.0': 'Standard',
        '20.0': 'Standard',
        '71.0': 'Non-Standard',
        '18.0': 'Standard',
        '24.0': 'Standard',
        '14.0': 'Standard',
        '74.0': 'Non-Standard',
        '15.0': 'Standard',  
        '77.0': 'Standard',
        '21.0': 'Standard',
        '72.0': 'Non-Standard',
        '27.0': 'Standard',
        '26.0': 'Standard',
        '75.0': 'Non-Standard',
        '96.0': 'Non-Standard',
        '0.0' : 'Non-Standard'
        }

df_norm.info()

print('''\n
<class 'pandas.core.frame.DataFrame'>
Int64Index: 75238 entries, 0 to 88622
Data columns (total 9 columns):
Nbhd                  75238 non-null int64
Sale_Price            75238 non-null int64
Total_Land_Size       75238 non-null float64
Total_Market_Value    75238 non-null int64
Total_SqFt            75238 non-null float64
Mkt_Land_Value        75238 non-null int64
Imp_Value             75238 non-null int64
House_Type            75238 non-null float64
House_Type_String     75238 non-null object
dtypes: float64(3), int64(5), object(1)    
\n''')


###############################################################################
#####  Item 1.12: Consolidation
###############################################################################

print('''\n
###############################################################################
#####  Stp 1.12: Consolidation
###############################################################################
\n''')    
print("\n Map dictionary values to the House_Type_String column.\n")

for key in House_Type_String_Dict:
    Replace = df_norm.loc[:,'House_Type_String'] == key
    df_norm.loc[Replace,'House_Type_String'] = House_Type_String_Dict[key]
    
print(df_norm['House_Type_String'].unique())  
# ['Standard' 'Non-Standard']

print(df_norm['House_Type_String'].value_counts())

print('''\n
Standard        65922
Non-Standard     9316
Name: House_Type_String, dtype: int64      
\n''')

df_norm.info()

###############################################################################
#####  Item 1.13: One-hot encoding
###############################################################################

print('''\n
###############################################################################
#####  Step 1.13: One-hot encoding
###############################################################################
\n''')      

# Create 2 new columns, one for each housing type in "House_Type_String"
df_norm.loc[:, "Standard"] = (df_norm.loc[:, "House_Type_String"] == "Standard").astype(int)
df_norm.loc[:, "Non-Standard"] = (df_norm.loc[:, "House_Type_String"] == "Non-Standard").astype(int)


df_norm.info()


###############################################################################
#####  Item 1.14: Normalization
###############################################################################

print('''\n
###############################################################################
#####  Step 1.14: Normalization
###############################################################################
\n''')      

############
# columns to scale with the standard scaler are:
# Sale_Price, Total_Land_Size, Total_Market_Value, Total_SqFt, Mkt_Land_Value,
# Imp_Value

df_scale_norm = df_norm

df_scale_norm = df_scale_norm.drop(columns=['House_Type', 'House_Type_String', 'Standard', 
                            'Non-Standard'])

df_norm.shape
# (75238, 11)   
df_scale_norm.shape    
# (75238, 7)


df_scale_norm.info()

print('''\n
df_scale_norm.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 75238 entries, 0 to 88622
Data columns (total 7 columns):
Nbhd                  75238 non-null int64
Sale_Price            75238 non-null int64
Total_Land_Size       75238 non-null float64
Total_Market_Value    75238 non-null int64
Total_SqFt            75238 non-null float64
Mkt_Land_Value        75238 non-null int64
Imp_Value             75238 non-null int64
dtypes: float64(2), int64(5)      
\n''')

scaler = StandardScaler()

print(scaler.fit(df_scale_norm))

StandardScaler()

print(scaler.mean_)

print('''\n
print(scaler.mean_)
StandardScaler(copy=True, with_mean=True, with_std=True)
[3.16801593e+06 7.97494622e+05 7.46993398e-01 5.40750942e+05
 1.74553017e+03 2.35519625e+05 3.05231317e+05]     
\n''')

# Use Standardscaler

print(scaler.transform(df_scale_norm))

df_scale_norm_transform = scaler.transform(df_scale_norm)



###############################################################################
#####  Item 2: Unsupervised Learning
###############################################################################

print('''\n
###############################################################################
#####  Step 2: Unsupervised Learning
###############################################################################
\n''')  
 
print('''\n  
###############################################################################
#####  Item 2.1:  Perform a K-Means with sklearn on some of your attributes.
#####               Include at least one categorical column and one numeric 
#####               attribute. Neither may be a proxy for the expert label 
#####               in supervised learning.
###############################################################################
\n''')


df_scale_norm.info()

print('''\n
df_scale_norm.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 75238 entries, 0 to 88622
Data columns (total 7 columns):
Nbhd                  75238 non-null int64
Sale_Price            75238 non-null int64
Total_Land_Size       75238 non-null float64
Total_Market_Value    75238 non-null int64
Total_SqFt            75238 non-null float64
Mkt_Land_Value        75238 non-null int64
Imp_Value             75238 non-null int64
dtypes: float64(2), int64(5)      
\n''')

# change to 10 clusters
kmeans_1 = KMeans(n_clusters=10, init='random', n_init=10, max_iter=600,
                tol=1e-04, random_state=0) 

kmeans_1.fit(np.array(df_scale_norm.astype(float)))
labels_1 = kmeans_1.predict(np.array(df_scale_norm.astype(float)))
# kmeans_1.fit(np.array(df_km.astype(float)))
# lables_1 = kmeans_1.predict(np.array(df_km.astype(float)))

print(labels_1)
len(labels_1) 
# 75238

counter_1 = Counter(labels_1)

print(counter_1)

print('''\n
print(counter_1)
Counter({2: 20137, 7: 19726, 3: 17016, 1: 16318, 6: 1322, 
5: 465, 0: 127, 9: 80, 4: 28, 8: 19})      
\n''')
    
###############################################################################    
#####   Item 2.2   Normalize the attributes prior to K-Means or justify why 
#####               you didn't normalize.
###############################################################################    

print('''\n
###############################################################################    
#####   Step 2.2   Normalize the attributes prior to K-Means or justify why 
#####               you didn't normalize.
###############################################################################       
\n''')

print('''\n
    Normalization was completed in Step 1.14  
\n''')

############################################################################### 
#####    Item 2.3   Add the cluster label to the data set to be used 
#####               in supervised learning    
###############################################################################     
    
print('''\n
############################################################################### 
#####    Step 2.3   Add the cluster label to the data set to be used 
#####               in supervised learning    
###############################################################################           
\n''')    

df_scale_norm['Clusters'] = pd.Series(labels_1, index=df_scale_norm.index)

df_scale_norm.info()
print("The Clusters column has been added to the dataset.")



df_scale_norm.describe()

print('''\n
The Clusters column has been added to the dataset.
 
               Nbhd    Sale_Price      ...          Imp_Value      Clusters
count  7.523800e+04  7.523800e+04      ...       7.523800e+04  75238.000000
mean   3.168016e+06  7.974946e+05      ...       3.052313e+05      3.415335
std    1.370015e+06  3.691147e+06      ...       1.400948e+06      2.323934
min    1.101006e+06  1.000000e+00      ...       0.000000e+00      0.000000
25%    2.207000e+06  2.940000e+05      ...       1.541000e+05      2.000000
50%    3.304001e+06  4.020000e+05      ...       2.215000e+05      3.000000
75%    4.303894e+06  5.500000e+05      ...       3.059000e+05      7.000000
max    6.101000e+06  1.455000e+08      ...       1.074198e+08      9.000000

[8 rows x 8 columns]     
\n''')

###############################################################################
#####  Item 3:  Supervised Learning
###############################################################################

print('''\n)
###############################################################################
#####  Step 3:  Supervised Learning
###############################################################################
\n''')  
 
###############################################################################
#####  Item 3.1: Ask a binary-choice question that describes your 
#####              classification. Write the question as a comment.
###############################################################################

print('''\n  
###############################################################################
#####  Step 3.1: Ask a binary-choice question that describes your 
#####              classification. Write the question as a comment.
#####      
#####              Is this a standard house?
###############################################################################
\n''')

# Add datat cloumns to df_scale_norm dataframe
df_scale_norm['House_Type'] = df_norm['House_Type']

df_scale_norm.info()

df_scale_norm['House_Type_String'] = df_norm['House_Type_String']

df_scale_norm.info()

df_scale_norm['Standard'] = df_norm['Standard']
df_scale_norm['Non-Standard'] = df_norm['Non-Standard']

df_scale_norm.info()

# Add type column with default zero entry

df_scale_norm['type'] = 0


# Now add 1 to the type column if it is a standard type

df_scale_norm.loc[(df_scale_norm['Standard'] == 1),'type'] = 1


###############################################################################
#####  Item 3.2: Split your data set into training and testing sets using 
#####              the proper function in sklearn.              
###############################################################################

print('''\n  
###############################################################################
#####  Step 3.2: Split your data set into training and testing sets using 
#####              the proper function in sklearn.
###############################################################################      
\n''')

df_scale_norm_in = df_scale_norm.drop(columns=['House_Type_String', 'Standard',
                                               'Non-Standard', 'House_Type'])
TestFraction = 0.33
MainSet = df_scale_norm_in

TrainSet, TestSet = train_test_split(MainSet, test_size=TestFraction)
print ('Test size should have been ', 
       TestFraction*len(MainSet), "; and is: ", len(TestSet))

TestSet.info()

###############################################################################
#####   Item 3.3: Use sklearn to train two classifiers on your training set, 
#####              like logistic regression and random forest.             
###############################################################################

print('''\n  
###############################################################################
#####  Step 3.3: Use sklearn to train two classifiers on your training set, 
#####              like logistic regression and random forest. 
###############################################################################      
\n''')


print ('\n Use logistic regression to predict house type.')
Target = "type"
Inputs = list(TrainSet.columns)
Inputs.remove(Target)
clf = LogisticRegression()
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

print ('\n Use random forest to predict house type.')
Target = "type"
Inputs = list(TrainSet.columns)
Inputs.remove(Target)
clf_1 = RandomForestClassifier()
clf_1.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])



print ('\n Use kneighbors classifier to predict house type.')
Target = "type"
Inputs = list(TrainSet.columns)
Inputs.remove(Target)
clf_2 = KNeighborsClassifier()
clf_2.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

###############################################################################
#####   Item 3.4: Apply your (trained) classifiers to the test set.       
###############################################################################

print('''\n  
###############################################################################
#####  Step 3.4: Apply your (trained) classifiers to the test set.  
###############################################################################      
\n''') 


# Logistic Regression
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]

# Random Forset
BothProbabilities_1 = clf_1.predict_proba(TestSet.loc[:,Inputs])
probabilities_1 = BothProbabilities_1[:,1]

# KNeighbors Classifier
BothProbabilities_2 = clf_2.predict_proba(TestSet.loc[:,Inputs])
probabilities_2 = BothProbabilities_2[:,1]


#############################################################################
##### Create Confusion Matrix Logistic Regression
#############################################################################

print ('\nConfusion Matrix and Metrics')

#############################################################################
#### Item 3.5: Create and present a confusion matrix for each classifier. 
#############################################################################


print('''\n  
###############################################################################
#####   Step 3.5: Create and present a confusion matrix for each classifier. 
###############################################################################      
\n''') 

#############################################################################
##### Logistic Regression
#############################################################################

#############################################################################
##### Set Probability Threshold = 0.5
#############################################################################

Threshold = 0.5 # Some number between 0 and 1 [Scaling the targt values]
print ("Probability Threshold is chosen to be:", Threshold)
predictions = (probabilities > Threshold).astype(int)
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print("Confusion Matrix for Logistic Regression")
print(CM)

print('''\n
Confusion Matrix for Logistic Regression
[[ 3056    23]
 [   22 21728]]
\n''')


#############################################################################
##### Random Forest
#############################################################################

#############################################################################
##### Set Probability Threshold = 0.5
#############################################################################
Threshold_1 = 0.5 # Some number between 0 and 1 [Scaling the targt values]
print ("Probability Threshold is chosen to be:", Threshold_1)
predictions_1 = (probabilities_1 > Threshold_1).astype(int)
CM_1 = confusion_matrix(TestSet.loc[:,Target], predictions_1)
tn_1, fp_1, fn_1, tp_1 = CM_1.ravel()
print("Confusion Matrix for Random Forest Classification")
print(CM_1)

print('''\n
Probability Threshold is chosen to be: 0.5
Confusion Matrix for Random Forest Classification
[[ 3055    30]
 [   27 21717]] 
\n''')

############################################################################
##### KNeighbors Classifier
#############################################################################

#############################################################################
##### Set Probability Threshold = 0.5
#############################################################################
Threshold_2 = 0.5 # Some number between 0 and 1 [Scaling the targt values]
print ("Probability Threshold is chosen to be:", Threshold_2)
predictions_2 = (probabilities_2 > Threshold_2).astype(int)
CM_2 = confusion_matrix(TestSet.loc[:,Target], predictions_2)
tn_2, fp_2, fn_2, tp_2 = CM_2.ravel()
print("Confusion Matrix for KNeighbors Classification")
print(CM_2)

print('''\n
Probability Threshold is chosen to be: 0.5
Confusion Matrix for KNeighbors Classification
[[ 2564   480]
 [  325 21460]]      
\n''')


#############################################################################
##### Item 3.6: Specify and justify your choice of probability threshold.
#############################################################################
print('''\n  
###############################################################################
#####   Step 3.6: Specify and justify your choice of probability threshold.
###############################################################################      
\n''') 

print(""""\n
      A threshold of 0.5 was chosen as the operational value because
      it works very well for the Logistic Regression approach and 
      also does not impact the Random Forest Classifier too much.
      
\n""")
#############################################################################
##### Item 3.7: For each classifier, create and present 2 accuracy metrics 
#####           based on the confusion matrix of the classifier.
#############################################################################

print('''\n  
###############################################################################
#####   Step 3.7: For each classifier, create and present 2 accuracy metrics 
#####              based on the confusion matrix of the classifier.
###############################################################################      
\n''') 


print("Logistic Regression -- Scores\n")
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(TestSet.loc[:,Target], probabilities)
AUC = auc(fpr, tpr)

print("AUC: ", np.round(AUC, 3))

print('''\n
Logistic Regression -- Scores

TP, TN, FP, FN: 21702 , 3024 , 20 , 83
Accuracy rate: 1.0
Precision: 1.0
Recall: 1.0
AUC:  0.996
\n''')


print("\nRandom Forest -- Scores\n")
print ("TP, TN, FP, FN:", tp_1, ",", tn_1, ",", fp_1, ",", fn_1)
AR_1 = accuracy_score(TestSet.loc[:,Target], predictions_1)
print ("Accuracy rate:", np.round(AR_1, 2))
P_1 = precision_score(TestSet.loc[:,Target], predictions_1)
print ("Precision:", np.round(P_1, 2))
R_1 = recall_score(TestSet.loc[:,Target], predictions_1)
print ("Recall:", np.round(R_1, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr_1, tpr_1, th_1 = roc_curve(TestSet.loc[:,Target], probabilities_1)
AUC_1 = auc(fpr_1, tpr_1)

print("AUC: ", np.round(AUC_1, 3))

print('''\n
 Random Forest -- Scores
 
TP, TN, FP, FN: 21755 , 3020 , 24 , 30
Accuracy rate: 1.0
Precision: 1.0
Recall: 1.0
AUC:  0.998 

\n''')


print("\nKNeighbors Classifier -- Scores\n")
print ("TP, TN, FP, FN:", tp_2, ",", tn_2, ",", fp_2, ",", fn_2)
AR_2 = accuracy_score(TestSet.loc[:,Target], predictions_2)
print ("Accuracy rate:", np.round(AR_2, 2))
P_2 = precision_score(TestSet.loc[:,Target], predictions_2)
print ("Precision:", np.round(P_2, 2))
R_2 = recall_score(TestSet.loc[:,Target], predictions_2)
print ("Recall:", np.round(R_2, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr_2, tpr_2, th_2 = roc_curve(TestSet.loc[:,Target], probabilities_2)
AUC_2 = auc(fpr_2, tpr_2)

print("AUC: ", np.round(AUC_2, 3))

print('''\n
  KNeighbors Classifier -- Scores

TP, TN, FP, FN: 21460 , 2564 , 480 , 325
Accuracy rate: 0.97
Precision: 0.98
Recall: 0.99
AUC:  0.966    
\n''')


#############################################################################
##### Item 3.8: For each classifier, calculate the ROC curve and it's AUC 
#####              using sklearn. Present the ROC curve. Present the AUC in 
#####              the ROC's plot.
#############################################################################

print('''\n  
###############################################################################
#####  Item 3.8: For each classifier, calculate the ROC curve and it's AUC 
#####              using sklearn. Present the ROC curve. Present the AUC in 
#####              the ROC's plot.
###############################################################################      
\n''') 

#############################################################################
##### Present Plot Logistic Regression
#############################################################################


print('''\n
#############################################################################
##### Present Plot Logistic Regression
#############################################################################      
\n''')
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('Logistic Regression ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--')
 # reference line for random classifier
plt.legend(loc="lower right")
plt.show()



############## Now for Random Forest

print('''\n
#############################################################################
##### Present Plot Random Forest Classifier
#############################################################################      
\n''')
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title(' Random Forest Classifier ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr_1, tpr_1, LW=3, label='ROC curve (AUC = %0.2f)' % AUC_1)
plt.plot([0, 1], [0, 1], color='red', LW=3, linestyle='--')
 # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

############## Nexrt is the KNeighbors Classifier

print('''\n
#############################################################################
##### Present Plot KNeighbors Classifier
#############################################################################      
\n''')
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('KNeighbors Classifier ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr_2, tpr_2, LW=3, label='ROC curve (AUC = %0.2f)' % AUC_2)
plt.plot([0, 1], [0, 1], color='green', LW=3, linestyle='--')
 # reference line for random classifier
plt.legend(loc="lower right")
plt.show()




###############################################################################
#####  Last Item: General Comments And Observations
###############################################################################


print('''\n
###############################################################################
#####  Last Step  General Comments And Observations
############################################################################### 


These classifiers are very good.

May be too good.  I checked the data and the process appears to be correct.

Also, the KNeighbors Classifier gives an AUC of 0.97.


That is an indication that the code works well.
    
''')

