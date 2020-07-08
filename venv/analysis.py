import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score

#read csv
data = pd.read_csv("C:/Users/User/PycharmProjects/invoke/surveyA.csv")

#get data info (how many columns
#data.info()
df = pd.DataFrame(data)

#remove special character in column names
df.rename(columns={'person_living_in_house': 'number of dependants', 'house_type': 'house type', 'house_value': 'house value','house_rental_fee': 'house rental fee','house_loan_pmt': 'house loan payment', 'transport_use': 'method of transportation', 'transport_spending': 'transport spending','public_transport_spending': 'public transport spending','house_utility': 'utilities','other_loan': 'other loan', 'education_loan': 'education loan','personal_loan': 'personal loan','kids_spending': 'kids spending','food_spending': 'food spending'}, inplace=True)
#print(df.columns)

#get how many null columns and drop them
print(df.isnull().sum())

#drop unrelated columns
df = df.drop(columns=['house value','age','race','gender','education'])

#drop columns with null value (size < 50)
df = df.dropna(subset=['salary', 'employment','married','number of dependants', 'house type', 'vehicle', 'method of transportation'])

#change null into the median value
df['house rental fee'] = df['house rental fee'].replace(np.NaN, df['house rental fee'].median())
df['house loan payment'] = df['house loan payment'].replace(np.NaN, df['house loan payment'].median())
df['transport spending'] = df['transport spending'].replace(np.NaN, df['transport spending'].median())
df['public transport spending'] = df['public transport spending'].replace(np.NaN, df['public transport spending'].median())
df['utilities'] = df['utilities'].replace(np.NaN, df['utilities'].median())
df['food spending'] = df['food spending'].replace(np.NaN, df['food spending'].median())
df['kids spending'] = df['kids spending'].replace(np.NaN, df['kids spending'].median())
df['personal loan'] = df['personal loan'].replace(np.NaN, df['personal loan'].median())
df['education loan'] = df['education loan'].replace(np.NaN, df['education loan'].median())
df['other loan'] = df['other loan'].replace(np.NaN, df['other loan'].median())
df['investment'] = df['investment'].replace(np.NaN, df['investment'].median())
df['savings1'] = df['savings1'].replace(np.NaN, df['savings1'].median())
df['savings2'] = df['savings2'].replace(np.NaN, df['savings2'].median())
df['savings3'] = df['savings3'].replace(np.NaN, df['savings3'].median())
df['savings4'] = df['savings4'].replace(np.NaN, df['savings4'].median())
df['savings5'] = df['savings5'].replace(np.NaN, df['savings5'].median())

#check if there is null value after cleaning the data
print(df.isnull().sum())

#change salary into two categories - <5k and >=5k
df['salary'] = df['salary'].replace('Less than 1K', '<5K')
df['salary'] = df['salary'].replace('1K to 2K', '<5K')
df['salary'] = df['salary'].replace('2K to 3K', '<5K')
df['salary'] = df['salary'].replace('3K to 4K', '<5K')
df['salary'] = df['salary'].replace('4K to 5K', '<5K')
df['salary'] = df['salary'].replace('5K to 6K', '>=5K')
df['salary'] = df['salary'].replace('7K to 8K', '>=5K')
df['salary'] = df['salary'].replace('8K to 9K', '>=5K')
df['salary'] = df['salary'].replace('9K to 10K', '>=5K')
df['salary'] = df['salary'].replace('10K or more ', '>=5K')

#convert to 2 categories -- 0 for <5K, 1 for >=5K
df['salarybi'] = df.apply(lambda row: 1 if '>=5K'in row['salary'] else 0, axis=1)
df = df.drop(['salary'], axis=1)

# Use one-hot encoding on categorial columns
df = pd.get_dummies(df, columns=['employment', 'married', 'house type', 'vehicle', 'method of transportation'])

# shuffle rows
df = df.sample(frac=1)

# split training and testing data
d_train = df[:1000]
d_test = df[1000:]
d_train_att = d_train.drop(['salarybi'], axis=1)
d_train_gt50 = d_train['salarybi']
d_test_att = d_test.drop(['salarybi'], axis=1)
d_test_gt50 = d_test['salarybi']
d_att = df.drop(['salarybi'], axis=1)
d_gt50 = df['salarybi']

# number of income > 5K in whole dataset:
print("Income >=5K: %d out of %d (%.2f%%)" % (np.sum(d_gt50), len(d_gt50), 100*float(np.sum(d_gt50)) / len(d_gt50)))

#get max depth for the decision tree
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_gt50, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std()*2))

#get score of the train/test
t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
t = t.fit(d_train_att, d_train_gt50)
t.score(d_test_att, d_test_gt50)
markah = t.score(d_test_att, d_test_gt50)

#print score for train/test
print('Score for train/test: (%0.4f)' % (markah))
scores = cross_val_score(t, d_att, d_gt50, cv=5)
# Show avarage score and +/- two standard deviations away (covering 95% or scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))

#get first row and convert to prediction test dataset
df.iloc[[0]].to_csv('prediction.csv', sep=',', encoding='utf-8', index=False)
# Prepare user profile
sample_df = pd.read_csv('prediction.csv', sep=',')
sample_df = sample_df.drop(['salarybi'], axis=1)
# Start predicting
predict_value = sample_df.iloc[0]
y_predict = t.predict([predict_value.tolist()])
y_predict[0] #0

#predict the salary for the test data
print('Prediction value (0 for > 5k, 1 for <= 5): ')
print(y_predict[0])

