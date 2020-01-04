import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

train["Sex"].loc[train["Sex"] == "male"] = 0
train["Sex"].loc[train["Sex"] == "female"] = 1
train["Embarked"].loc[train["Embarked"] == "S" ] = 0
train["Embarked"].loc[train["Embarked"] == "C" ] = 1
train["Embarked"].loc[train["Embarked"] == "Q"] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"].loc[test["Sex"] == "male"] = 0
test["Sex"].loc[test["Sex"] == "female"] = 1
test["Embarked"].loc[test["Embarked"] == "S"] = 0
test["Embarked"].loc[test["Embarked"] == "C"] = 1
test["Embarked"].loc[test["Embarked"] == "Q"] = 2
test.Fare.loc[152] = test.Fare.median()

# scikit-learnのインポートをします
from sklearn import tree

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
# 追加となった項目も含めて予測モデルその2で使う値を取り出す
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
# 決定木の作成とアーギュメントの設定
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
# tsetから「その2」で使う項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
# 「その2」の決定木を使って予測をしてCSVへ書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ["Survived"])
my_solution_tree_two.to_csv("./data/submit.csv", index_label = ["PassengerId"])
