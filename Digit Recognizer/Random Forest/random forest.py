from data_loader import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ml = RandomForestClassifier(criterion='entropy', n_estimators=1500, n_jobs=4)

num, image = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(image, num, test_size=0.3, random_state=0)

ml.fit(X_train, Y_train)
Y_pred = ml.predict(X_test)

print("%.2f" % accuracy_score(Y_test, Y_pred))