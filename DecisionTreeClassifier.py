
import pandas as pd
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV 
from sklearn.svm import SVR 
from sklearn.feature_selection import SelectKBest , chi2
from sklearn.metrics import accuracy_score
features_names = ['times_pregnant','Plasma_glucose_concentration_2hr',
                  'blood_pressure','Triceps_skin_fold_thickness','Hr2_serum_insulin','BOI','Diabetes_pedigree_function','Age','Class']
db = pd.read_csv('diabetes.csv',header=0,names=features_names)
array = db.values
x = array[:,0:8]
y = array[:,8]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)
model = DecisionTreeClassifier()

clf = model.fit(x_train,y_train) #Training Data 
y_pred = model.predict(x_test)    #Accepted Data 
print("%s:%f" % ('The accuracy score before Applying any Features',accuracy_score(y_test,y_pred))) #Compare between Training Data & Accepted Data 
h = SelectKBest(chi2 , k=4) #instance from SelectKBest model with k = 4
xfeature = h.fit_transform(x_train,y_train) #Testing Data for Exact the Best features 
print("The Best Features that influnes on the data are (using Univariate Feature Selection): " , [h.get_support(indices=True)])
x_newUnivariate = array[:,[1,4,5,7]] #x new after applaying Univariate Feature
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_newUnivariate, y, test_size=0.2, random_state=6)
clf = model.fit(x_train1,y_train1)
y_pred1 = model.predict(x_test1)
UnivariateAccuracyScore = accuracy_score(y_test1,y_pred1)
print("%s:%f" % ('The accuracy score after Univariate Feature Selection',UnivariateAccuracyScore))


estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv = 7)
selector = selector.fit(x_train,y_train)
print("The Best Features that influnes on the data are (using Recursive Feature Elimination ):", selector.ranking_)

x_newRecursive = array[:,[1,2,3,4,5]] #x new after applaying Recursive Feature
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_newRecursive, y, test_size=0.2, random_state=6)
clf = model.fit(x_train2,y_train2)
y_pred2 = model.predict(x_test2)
recursicefeatureAccuracyScore = accuracy_score(y_test2,y_pred2)
print("%s:%f" % ('The accuracy score After Recursive Feature Elimination ',recursicefeatureAccuracyScore))


param_grid = {'max_depth': range(1,5),'min_samples_split': range(2,10) , 'min_samples_leaf':range(1,10)}  
kfold = KFold(n_splits=7 , random_state=4) 
model = GridSearchCV(model , param_grid, cv= kfold , scoring='accuracy')
model.fit(x_train,y_train)

print("The best parameters ..:")
print(model.best_params_)
print("The best score ..:")
print(model.best_score_)

if (UnivariateAccuracyScore > recursicefeatureAccuracyScore ): #Testing purpose for high performance
    x_new_data = x_train1
    x_test_data = x_test1
    y_test_data = y_test1
    
else:
    x_new_data = x_train2
    x_test_data = x_test2
    y_test_data = y_test2

params={'max_depth':2, 'min_samples_split':2,'min_samples_leaf':1}
DT = DecisionTreeClassifier(**params)
cv_result = model_selection.cross_val_score(DT , x_new_data , y_train2 , cv=kfold,scoring='accuracy')
msg = '%s: %f (%f)' % ('Estimated performance: ', cv_result.mean() , cv_result.std())
print(msg)
DT.fit(x_new_data , y_train2)
prediction = DT.predict(x_test_data)
print('%s : %f' % ('Test Performance After Applying Hyper Parameter Tunning : ' , accuracy_score(y_test_data,prediction)))


