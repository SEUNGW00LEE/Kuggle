# 필요한 library 불러오기

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
rc('font', family='AppleGothic')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.loc[train.지역.isin(['경기도','서울특별시']), '지역'] = '수도권(서울/경기)'
train.loc[train.지역.isin(['경상남도','울산광역시','부산광역시','대구광역시','경상북도']), '지역'] = '경상도'
train.loc[train.지역.isin(['전라남도','광주광역시','전라북도']), '지역'] = '전라도'
train.loc[train.지역.isin(['충청남도','충청북도','대전광역시','세종특별자치시']), '지역'] = '충청도'

test.loc[test.지역.isin(['경기도','서울특별시']), '지역'] = '수도권(서울/경기)'
test.loc[test.지역.isin(['경상남도','울산광역시','부산광역시','대구광역시','경상북도']), '지역'] = '경상도'
test.loc[test.지역.isin(['전라남도','광주광역시','전라북도']), '지역'] = '전라도'
test.loc[test.지역.isin(['충청남도','충청북도','대전광역시','세종특별자치시']), '지역'] = '충청도'


train.rename(columns={'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철수',
                      '도보 10분거리 내 버스정류장 수': '버스수'}, inplace=True)
test.rename(columns={'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철수',
                     '도보 10분거리 내 버스정류장 수': '버스수'}, inplace=True)

for col in ['임대료', '임대보증금']:
    train[col] = train[col].replace('-', np.nan).astype(float)
    test[col] = test[col].replace('-', np.nan).astype(float)

train['임대료'] = train['임대료'].astype(float)
train['임대보증금'] = train['임대보증금'].astype(float)

test['임대료'] = test['임대료'].astype(float)
test['임대보증금'] = test['임대보증금'].astype(float)

nan_subway = train[(train['지역'] == '대전광역시') & (train['지하철수'].isna())]
nan_subway_codes = nan_subway['단지코드'].unique()

train['버스수'].fillna(10, inplace=True)
train['지하철수'].fillna(0, inplace=True)

nan_subway = test[(test['지역'] == '대전광역시') & (train['지하철수'].isna())]

test['지하철수'].fillna(0, inplace=True)
test[test['단지코드']=='C2411'] 
test[test['단지코드']=='C2411']['자격유형'].fillna('A')

test[test['단지코드']=='C2253'] 
test[test['단지코드']=='C2253']['자격유형'].fillna('C')

missing_data = train[train['임대료'].isna() | train['임대보증금'].isna()]
train.loc[train['공급유형'] == '장기전세', '임대료'] = 0
test.loc[test['공급유형'] == '장기전세', '임대료'] = 0

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_non_store = train[train['공급유형'] != '상가']

regions = train_non_store['지역'].unique()

predicted_rent = [] 
predicted_deposit = [] 

for region in regions:
    region_data = train_non_store[train_non_store['지역'] == region].copy()



    X = region_data[~region_data['임대료'].isna()][['단지내주차면수']]
    y = region_data[~region_data['임대료'].isna()]['임대료']
    if len(X) > 0:
        model_rent = LinearRegression()
        model_rent.fit(X, y)


        X_missing_rent = region_data[region_data['임대료'].isna()][['단지내주차면수']]
        if len(X_missing_rent) > 0:
            pred_rent = model_rent.predict(X_missing_rent)
            predicted_rent.extend(pred_rent)

    X = region_data[~region_data['임대보증금'].isna()][['단지내주차면수']]
    y = region_data[~region_data['임대보증금'].isna()]['임대보증금']
    if len(X) > 0:
        model_deposit = LinearRegression()
        model_deposit.fit(X, y)
        
        X_missing_deposit = region_data[region_data['임대보증금'].isna()][['단지내주차면수']]
        
        if len(X_missing_deposit) > 0:
            pred_deposit = model_deposit.predict(X_missing_deposit)
            predicted_deposit.extend(pred_deposit)

missing_rent_indices = train_non_store[train_non_store['임대료'].isna()].index
for idx, value in zip(missing_rent_indices, predicted_rent):
    train.at[idx, '임대료'] = value

missing_deposit_indices = train_non_store[train_non_store['임대보증금'].isna()].index
for idx, value in zip(missing_deposit_indices, predicted_deposit):
    train.at[idx, '임대보증금'] = value

missing_values = train[train['공급유형'] != '상가'][['임대료', '임대보증금']].isnull().sum()


test_non_store = test[test['공급유형'] != '상가']

regions = test_non_store['지역'].unique()


predicted_rent = [] 
predicted_deposit = []

# 지역을 기준으로 반복문 돌림
for region in regions:
    region_data = test_non_store[test_non_store['지역'] == region].copy()

    # 임대료에 대한 회귀 모델 학습
    X = region_data[~region_data['임대료'].isna()][['단지내주차면수']]
    y = region_data[~region_data['임대료'].isna()]['임대료']
    if len(X) > 0:
        model_rent = LinearRegression()
        model_rent.fit(X, y)

        X_missing_rent = region_data[region_data['임대료'].isna()][['단지내주차면수']]

        if len(X_missing_rent) > 0:
            pred_rent = model_rent.predict(X_missing_rent)
            predicted_rent.extend(pred_rent)


    X = region_data[~region_data['임대보증금'].isna()][['단지내주차면수']]
    y = region_data[~region_data['임대보증금'].isna()]['임대보증금']
    if len(X) > 0:
        model_deposit = LinearRegression()
        model_deposit.fit(X, y)

        X_missing_deposit = region_data[region_data['임대보증금'].isna()][['단지내주차면수']]

        if len(X_missing_deposit) > 0:
            pred_deposit = model_deposit.predict(X_missing_deposit)
            predicted_deposit.extend(pred_deposit)


missing_rent_indices = test_non_store[test_non_store['임대료'].isna()].index
for idx, value in zip(missing_rent_indices, predicted_rent):
    test.at[idx, '임대료'] = value

missing_deposit_indices = test_non_store[test_non_store['임대보증금'].isna()].index
for idx, value in zip(missing_deposit_indices, predicted_deposit):
    test.at[idx, '임대보증금'] = value

missing_values = test[test['공급유형'] != '상가'][['임대료', '임대보증금']].isnull().sum()

train.loc[train.공급유형.isin(['공공임대(5년)', '공공분양', '공공임대(10년)', '공공임대(분납)','공공임대(50년)']), '공급유형'] = '공공임대(5년/10년/분납/분양)'
test.loc[test.공급유형.isin(['공공임대(5년)', '공공분양', '공공임대(10년)', '공공임대(분납)','공공임대(50년)']), '공급유형'] = '공공임대(5년/10년/분납/분양)'

train.loc[train.공급유형.isin(['장기전세', '국민임대']), '공급유형'] = '국민임대/장기전세'
test.loc[test.공급유형.isin(['장기전세', '국민임대']), '공급유형'] = '국민임대/장기전세'

train['자격유형'] = train['자격유형'].apply(lambda x: 'A-B' if x in ['A', 'B'] else x)
test['자격유형'] = train['자격유형'].apply(lambda x: 'A-B' if x in ['A', 'B'] else x)

train['자격유형'] = train['자격유형'].apply(lambda x: 'E-F' if x in ['E', 'F'] else x)
test['자격유형'] = train['자격유형'].apply(lambda x: 'E-F' if x in ['E', 'F'] else x)

train['자격유형'] = train['자격유형'].apply(lambda x: 'N-O' if x in ['N', 'O'] else x)
test['자격유형'] = train['자격유형'].apply(lambda x: 'N-O' if x in ['N', 'O'] else x)


object_columns = train.select_dtypes(include=['object']).columns.tolist()
object_columns.remove('단지코드') 

train_encoded = pd.get_dummies(train, columns=object_columns)
test_encoded = pd.get_dummies(test, columns=object_columns)

train_store_data = train_encoded[train_encoded['임대건물구분_상가'] == 1].drop(columns=['임대료', '임대보증금'])
train_non_store_data = train_encoded[train_encoded['임대건물구분_상가'] != 1]

test_store_data = test_encoded[test_encoded['임대건물구분_상가'] == 1].drop(columns=['임대료', '임대보증금'])
test_non_store_data = test_encoded[test_encoded['임대건물구분_상가'] != 1]

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# List to store results for comparison
results = []

# Define the parameter grid for XGBoost
param_grid = {'max_depth' : np.arange(3,9,1) ,
                  "n_estimators": [100],
                  'min_child_weight' : np.arange(1, 8, 1), 
                  'gamma' : [0,1,2,3],
                  "learning_rate": [0.015,0.02,0.025, 0.03, 0.035, 0.04, 0.05 ],
                  'subsample' :np.arange(0.8, 1.0, 0.1)}

# Define datasets
datasets = {
    '상가': (train_store_data, test_store_data),
    '비상가': (train_non_store_data, test_non_store_data)
}

# Define scalers
scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler()
}

# Loop through datasets (상가 and 비상가)
for dataset_name, (train_data, test_data) in datasets.items():
    
    # Split data
    X_train = train_data.drop(columns=['단지코드', '등록차량수'])
    y_train = train_data['등록차량수']
    X_test = test_data.drop(columns=['단지코드'])
    
    # Loop through scalers
    for scaler_name, scaler in scalers.items():
        
        # Normalize data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Grid Search with XGBoost
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'scaler': scaler_name,
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_
        })

results_df = pd.DataFrame(results)
results_df
