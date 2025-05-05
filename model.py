from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# 1. Iris データセットをロード
iris = datasets.load_iris()

# 2. 特徴量とラベルを準備
features = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

# 3. モデルの作成と学習
model = RandomForestClassifier()
model.fit(features, target)

# 4. モデルを保存
with open("model_iris.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ モデルの学習と保存が完了しました。")
