import numpy as np 
import pandas as pd

# data = pd.read_csv('Bengaluru_House_Data.csv')
# # print(data)

# # Preprocess the data by handling missing values:

# data = data.drop(columns=['area_type', 'availability', 'society', 'balcony'])
# # print(data)

# data['location'].fillna('unknown', inplace=True)
# data['size'].fillna(data['size'].mode()[0], inplace=True)
# data['bath'].fillna(data['bath'].median(), inplace=True)


# def convert_sqrt(x):
#     try:
#         if '-' in str(x):
#             a, b = x.split('-')
#             return (float(a) + float(b)) / 2
#         return float(x)
#     except:
#         return None

# data['total_sqft'] = data['total_sqft'].apply(convert_sqrt)
# data = data.dropna(subset=['total_sqft'])

# # print(data.isnull().sum())
# # print(data)

# # encoding categorical variables:

# def extract_num(x):
#     try:
#         return int(str(x).split(" ")[0])   # "2 BHK" -> 2 , "4 Bedroom" -> 4
#     except:
#         return None

# data["BHK_or_Bedroom"] = data["size"].apply(extract_num)
# data = data.drop(columns=["size"])

# # yeh har ghar ka rooms ka square feet ko dekh raha ha khun ka kuch ghar ka squareF
# # bhot ajeeb hote hain or wo ghar marketing ka leye issue karte hain
# data = data[(data['total_sqft']/data['BHK_or_Bedroom']) < 1000]  

# data['price_per_sqft'] = data['price'] / data['total_sqft']

# data = pd.get_dummies(data, columns=['location'], drop_first=True)
# # print(data)

# data.to_csv('Cleaned_House_Pridict_Data.csv', index=False)
# print('Data clean kar ka save kar diya ha: Cleaned_House_Pridict_Data.csv')

data = pd.read_csv('Cleaned_House_Pridict_Data.csv')

# scaling the data:

from sklearn.preprocessing import MinMaxScaler

x = data.drop(columns=['price'])     
y = data['price']                    # yeh model ka leye ha 

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
# print(x_scaled[:5])


# Model Development:

from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
# model = LinearRegression()
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(x_train, y_train)

y_pridict = model.predict(x_test)

#  Model Evaluation:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pridict)
mse = mean_squared_error(y_test, y_pridict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pridict)

print('Model average kitna galat predict kar raha hai: ', mae)
print('Agar model mein zaida error howe to yeh hame bata de ga: ', mse)
print('Yeh bhi large errors ko punish karta hai: ', rmse)
print('Overall accuracy batata ha: ',r2 )

