from HelperFunctions import *
from sklearn import preprocessing
train_feature, train_labels = format_data("/Users/yazen/Desktop/datasets/Titanic/train.csv")
test_feature = pd.read_csv("/Users/yazen/Desktop/datasets/Titanic/test.csv")
separator = train_feature.shape[0]

newFeature = pd.concat((train_feature, test_feature), axis=0)

objects = newFeature.select_dtypes(include=["object"])
numeric = newFeature.select_dtypes(exclude=["object"])
columns = numeric.columns
numeric = numeric.fillna(0)
scalar = preprocessing.MinMaxScaler()
numeric = scalar.fit_transform(numeric)
numeric = pd.DataFrame(numeric, columns= columns)

objects = pd.get_dummies(objects)

train_feature_objects = objects[:separator]
test_feature_objects = objects[separator:]

train_feature_numeric = numeric[:separator]
test_feature_numeric = numeric[separator:]

train_feature = pd.concat([train_feature_objects, train_feature_numeric], axis=1)
test_feature = pd.concat([test_feature_objects, test_feature_numeric], axis=1)

dev_feature = train_feature[700:]
train_feature = train_feature[:700]

dev_labels = train_labels[700:]
train_labels = train_labels[:700]
