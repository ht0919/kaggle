from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# 前処理：欠損データの補完
# 代理データを中央値や最頻値で置き換える
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# 前処理：カテゴリカルデータの数値化
# Sexは['male','female']を[0,1]に、Embarkedは['S','C','Q']を[0,1,2]に変換
train["Sex"].loc[train["Sex"] == "male"] = 0
train["Sex"].loc[train["Sex"] == "female"] = 1
train["Embarked"].loc[train["Embarked"] == "S"] = 0
train["Embarked"].loc[train["Embarked"] == "C"] = 1
train["Embarked"].loc[train["Embarked"] == "Q"] = 2

# testもtrainと同様に前処理
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"].loc[test["Sex"] == "male"] = 0
test["Sex"].loc[test["Sex"] == "female"] = 1
test["Embarked"].loc[test["Embarked"] == "S"] = 0
test["Embarked"].loc[test["Embarked"] == "C"] = 1
test["Embarked"].loc[test["Embarked"] == "Q"] = 2
test.Fare.loc[152] = test.Fare.median()

#　家族数を追加
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
test_two = test.copy()
test_two["family_size"] = test_two["SibSp"] + test_two["Parch"] + 1

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features = train_two[["Pclass", "Age", "Sex", "Fare", "family_size", "Embarked"]].values

# モデルは勾配ブースティング
forest = GradientBoostingClassifier(n_estimators=55, random_state=9)
forest = forest.fit(features, target)

# testから値を取り出す
test_features = test_two[["Pclass", "Age", "Sex", "Fare", "family_size", "Embarked"]].values

# 予測をしてCSVへ書き出す
my_prediction_forest = forest.predict(test_features)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_forest = pd.DataFrame(my_prediction_forest, PassengerId, columns=["Survived"])
my_solution_forest.to_csv("./data/submit.csv", index_label=["PassengerId"])
