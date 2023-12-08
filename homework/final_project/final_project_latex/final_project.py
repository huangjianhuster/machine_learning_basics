import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


### Load dataset
train_csv = "../train.csv"
test_csv = "../test.csv"
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

### Statistics of the training dataset
training_samples = train_df.shape[0]
print(f"shape of training set: {str(train_df.shape)}")
print(f"shape of testing set: {str(test_df.shape)}")

def get_statistics(df):
    cols = df.columns.values
    print(f"{'col_name':<12}{'total_len':<12}{'unique_num':<12}{'nan_number':<12}{'no_nan_number':<12}")
    feature_dict = {}
    for col in cols: # np.delete(cols, [0, 3, 8]):
        print(f"{col:<12}{len(df[col]):<12}{len(df[~df[col].isna()][col].unique()):<12}{df[col].isnull().sum():<12}{len(df[col]) - df[col].isnull().sum():<12}")
        feature_dict[col] = df[~df[col].isna()][col].unique()
    return feature_dict

print("\n### Statistics of the training dataset ###")
training_features_unique = get_statistics(train_df)
print("\n### Statistics of the testing dataset ###")
training_features_unique = get_statistics(test_df)

### Data cleaning
def cleaning_dataset(df_raw, impute=True, remove_missing=True):
    # Drop unnecessary columns
    df_raw = df_raw.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # transform [Sex] to binary
    df_raw['Sex'] = df_raw['Sex'].map({'male': 0, 'female': 1})
    
    # Decide on whether remove the two missing "Embarked" samples
    if remove_missing:
        missing_embark_idx = df_raw.index[df_raw['Embarked'].isnull()]
        df_raw.drop(missing_embark_idx, inplace=True)
    
    # One-hot encoding for [Embarked]
    df = pd.get_dummies(df_raw, columns=['Embarked'], prefix='Embarked', dtype='int')
    
    # Imputation using median
    if impute:
        for i in df.columns:
            df[i] = df[i].fillna(df[i].median())

    # Normalization: MinMaxScaler
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    df_normalized[['Fare', 'Age']] = scaler.fit_transform(df_normalized[['Fare', 'Age']])
    return df, df_normalized

# combine training and testing data <-- for statistics and also for imputation
df_all = pd.concat([train_df.drop(["Survived"], axis=1), test_df], ignore_index=True)
df_all_cleaned, df_all_cleaned_normalized = cleaning_dataset(df_all, impute=True, remove_missing=False)

print("\n### Statistics of the final cleaned dataset (merged training and testing) ###")
final_features_unique = get_statistics(df_all_cleaned_normalized)
print(df_all_cleaned_normalized.describe())

### Imputation plot
print("\n### Imputation of [Age]")

bins = 16
bin_range = (df_all['Age'].min(), df_all['Age'].max() )
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(df_all['Age'], bins=bins, range=bin_range, alpha=0.7, color='skyblue', edgecolor='gray')
axs[0].set_title("Original", fontsize=16)
axs[1].hist(df_all_cleaned['Age'], bins=bins, range=bin_range, alpha=0.7, color='palegreen', edgecolor='gray')
axs[1].set_title("Imputation with the average age", fontsize=16)
for i in axs.flatten():
    i.set_xlabel("Age", fontsize=16)
    i.set_ylabel("Occurance", fontsize=16)
    i.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("./imputation_age.png")
plt.show()

### Generate Training dataset and Test set
train_final_df = df_all_cleaned_normalized[:training_samples]
testing_final_df = df_all_cleaned_normalized[training_samples:]

X = train_final_df
y = train_df["Survived"]


#################################################################################################################
### Hyperparameter searching for Random forest
print("\n### Hyperparameter searching & model training (Random forest) ###")
# try more random seed if you want
# random_seed_list = [11, 37, 73, 89, 149]
random_seed_list = [11, ]
result_summary_rf = []

for random_seed in random_seed_list:
    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, ],
        'max_depth': [None, 2, 5, ],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a random forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    results_df_rf = pd.DataFrame(grid_search.cv_results_)[['param_max_depth', 'param_min_samples_leaf', \
                                                           'param_min_samples_split', 'param_n_estimators',\
                                                           'mean_test_score', 'std_test_score']]
    results_df_rf['Seed'] = results_df_rf.shape[0]*[random_seed, ]
    column_order = ['Seed'] + [col for col in results_df_rf.columns if col != "Seed" ]
    results_df_rf = results_df_rf[column_order]
    
    # Access the detailed results
    cv_results = grid_search.cv_results_
    
    # iterate through all models
    all_models = grid_search.cv_results_['params']
    test_acc = []
    for model_params in all_models:
        model = RandomForestClassifier(**model_params, random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"{model_params}, accuracy: {accuracy}")
        test_acc.append(accuracy)
    results_df_rf["testing_accuracy"] = test_acc
    result_summary_rf.append(results_df_rf)

# Merge all results from different seeds
final_results_rf = pd.concat(result_summary_rf, ignore_index=True)
final_results_rf = final_results_rf.sort_values(by="testing_accuracy", ascending=False)
final_results_rf.to_csv("RF_summary.csv", index=False)

# Print result and plot
print("\n### Final results of Random forest")
best_params_rf = final_results_rf.iloc[0, :]
print("The best parameters and results: \n", best_params_rf)

### Train the final model using the best params
final_hyper_params = {
    'max_depth': best_params_rf["param_max_depth"],
    'min_samples_leaf': best_params_rf["param_min_samples_leaf"],
    'min_samples_split': best_params_rf["param_min_samples_split"],
    'n_estimators': best_params_rf["param_n_estimators"]
}
model = RandomForestClassifier(**final_hyper_params, random_state=best_params_rf['Seed'])
model.fit(X, y)
y_pred_submit = model.predict(testing_final_df)
df_submit = pd.DataFrame(columns=["PassengerId", "Survived"])
df_submit["PassengerId"] = testing_final_df.index + 1
df_submit["Survived"] = y_pred_submit
df_submit.to_csv("RF_submit.csv", index=False)
print("\n### Final prediction from the Randomforest model")
print(df_submit)

### Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
colors = np.where(final_results_rf['testing_accuracy']< 0.83, 'red', \
                  np.where((final_results_rf['testing_accuracy'] >= 0.83) & (final_results_rf['testing_accuracy'] < 0.87),\
                            'orange', 'green'))
ax[0].scatter(final_results_rf.index, final_results_rf['testing_accuracy'], c=colors)
ax[0].set_title("Random Forest", fontsize=14)
ax[0].set_xlabel("Model index", fontsize=14)
ax[0].set_ylabel("Testing Accuracy", fontsize=14)
ax[0].tick_params(axis='both', which='major', labelsize=14)
ax[0].axhline(0.83, linestyle="--", color="k")
ax[0].axhline(0.87, linestyle="--", color="k")

ax[1].barh(importance_df_rf['Feature'], importance_df_rf['Importance'])
ax[1].set_xlabel('Importance', fontsize=14)
ax[1].set_title('Feature Importances in Random Forest', fontsize=14)
ax[1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig("./RF.png")
plt.show()

print("\n### End of the Random forest training ###")



#################################################################################################################
### Hyperparameter searching for fixed-shape universal approximator (Kernel method)
print("\n### Hyperparameter searching & model training (fixed-shape kernel method) ###")
X = train_final_df
y = train_df["Survived"]
# try more random seed if you want
# random_seed_list = [11, 37, 73, 89, 149]
random_seed_list = [11, ]
result_summary_svm = []

for random_seed in random_seed_list:
    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization Parameter
        'kernel': ['rbf'],
        # 'degree': [2, 3],  # Only relevant for 'poly' kernel
        'gamma': ['scale', 'auto', 0.1, 1]  # Only relevant for 'rbf' kernel
    }

    # Create a SVM classifier
    svm_classifier = SVC(random_state=random_seed)

    # Create a GridSearchCV object
    grid_search_svm = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')

    # Fit the GridSearchCV object to the training data
    grid_search_svm.fit(X_train, y_train)

    results_df_svm = pd.DataFrame(grid_search_svm.cv_results_)[['param_C', 'param_gamma',\
                                                                'param_kernel', 'mean_test_score', 'std_test_score']]
    results_df_svm['Seed'] = results_df_svm.shape[0]*[random_seed, ]
    column_order = ['Seed'] + [col for col in results_df_svm.columns if col != "Seed" ]
    results_df_svm = results_df_svm[column_order]

    all_models = grid_search_svm.cv_results_['params']
    test_acc = []
    for model_params in all_models:
        model = SVC(**model_params, random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"{model_params}, accuracy: {accuracy}")
        test_acc.append(accuracy)
    results_df_svm["testing_accuracy"] = test_acc
    result_summary_svm.append(results_df_svm)

final_results_svm = pd.concat(result_summary_svm, ignore_index=True)
final_results_svm = final_results_svm.sort_values(by="testing_accuracy", ascending=False)
final_results_svm.to_csv("SVM_summary.csv", index=False)

# Print result and plot
print("\n### Final results of SVM kernel method")
best_params_svm = final_results_svm.iloc[0, :]
print("The best parameters and results: \n", best_params_svm)

### Train the final model using the best params
final_hyper_params_svm = {
    'C': best_params_svm["param_C"],
    'gamma': best_params_svm["param_gamma"],
    'kernel': best_params_svm["param_kernel"],
}
model_svc = SVC(**final_hyper_params_svm, random_state=best_params_svm['Seed'])
model_svc.fit(X, y)
y_pred_svm_submit = model_svc.predict(testing_final_df)
df_svm_submit = pd.DataFrame(columns=["PassengerId", "Survived"])
df_svm_submit["PassengerId"] = testing_final_df.index+1
df_svm_submit["Survived"] = y_pred_svm_submit
df_svm_submit.to_csv("SVM_submit.csv", index=False)
print("\n### Final prediction from the SVM kernel model")
print(df_svm_submit)

### Perform permutation importance
result = permutation_importance(model_svc, X, y, n_repeats=30, random_state=best_params_svm['Seed'])
importance_svm = result.importances_mean
feature_names_svm = X.columns
importance_df_svm = pd.DataFrame({'Feature': feature_names_svm, 'Importance': importance_svm})
importance_df_svm = importance_df_svm.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
colors = np.where(final_results_svm['testing_accuracy']< 0.83, 'red', \
                  np.where((final_results_svm['testing_accuracy'] >= 0.83) & (final_results_svm['testing_accuracy'] < 0.87),\
                            'orange', 'green'))
ax[0].scatter(final_results_svm.index, final_results_svm['testing_accuracy'], c=colors)
ax[0].set_title("SVM Kernel", fontsize=14)
ax[0].set_xlabel("Model index", fontsize=14)
ax[0].set_ylabel("Testing Accuracy", fontsize=14)
ax[0].tick_params(axis='both', which='major', labelsize=14)
ax[0].axhline(0.83, linestyle="--", color="k")
ax[0].axhline(0.87, linestyle="--", color="k")

ax[1].barh(importance_df_svm['Feature'], importance_df_svm['Importance'])
ax[1].set_xlabel('Importance', fontsize=14)
ax[1].set_title('Feature Importances in SVM Kernel', fontsize=14)
ax[1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig("./SVM.png")
plt.show()






#################################################################################################################
### Hyperparameter searching for neural network-based model
print("\n### Hyperparameter searching & model training (Neural Network) ###")
X = train_final_df
y = train_df["Survived"]
# try more random seed if you want
# random_seed_list = [11, 37, 73, 89, 149]
random_seed_list = [11, ]
result_summary_nn = []

for random_seed in random_seed_list:
    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Create a neural network classifier
    nn_classifier = MLPClassifier(random_state=random_seed)

    # Cross-validation to search for optimal hyperparameters
    param_grid = {
        'hidden_layer_sizes': [(10,), (15,), (20,), (5, 10)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.1, 0.01],
        'max_iter': [500],
    }

    # Using 'accuracy' as the scoring metric, adjust as needed
    grid_search_nn = GridSearchCV(nn_classifier, param_grid, cv=5, scoring='accuracy')
        # Fit the GridSearchCV object to the training data
        
    grid_search_nn.fit(X_train, y_train)

    results_df_nn = pd.DataFrame(grid_search_nn.cv_results_)[['param_alpha', 'param_hidden_layer_sizes', \
                                                           'param_learning_rate_init', 'param_max_iter',\
                                                           'mean_test_score', 'std_test_score']]
    results_df_nn['Seed'] = results_df_nn.shape[0]*[random_seed, ]
    column_order = ['Seed'] + [col for col in results_df_nn.columns if col != "Seed" ]
    results_df_nn = results_df_nn[column_order]
    # results_df_rf = results_df_rf.sort_values(by="mean_test_score", ascending=False)
    
    # iterate through all models
    all_models = grid_search_nn.cv_results_['params']
    test_acc = []
    for model_params in all_models:
        model = MLPClassifier(**model_params, random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"{model_params}, accuracy: {accuracy}")
        test_acc.append(accuracy)
    results_df_nn["testing_accuracy"] = test_acc
    result_summary_nn.append(results_df_nn)

final_results_nn = pd.concat(result_summary_nn, ignore_index=True)
final_results_nn = final_results_nn.sort_values(by="testing_accuracy", ascending=False)
final_results_nn.to_csv("NN_summary.csv", index=False)

# Print result and plot
print("\n### Final results of Neural Network method")
best_params_nn = final_results_nn.iloc[0, :]
print("The best parameters and results: \n", best_params_nn)

### Train the final model using the best params
final_hyper_params_nn = {
    'alpha': best_params_nn["param_alpha"],
    'hidden_layer_sizes': best_params_nn['param_hidden_layer_sizes'],  
    'learning_rate_init': best_params_nn['param_learning_rate_init'],
    'max_iter': best_params_nn['param_max_iter']
}
model_nn = MLPClassifier(**final_hyper_params_nn, random_state=best_params_nn['Seed'])
model_nn.fit(X, y)
y_pred_nn_submit = model_nn.predict(testing_final_df)
df_nn_submit = pd.DataFrame(columns=["PassengerId", "Survived"])
df_nn_submit["PassengerId"] = testing_final_df.index+1
df_nn_submit["Survived"] = y_pred_nn_submit
df_nn_submit.to_csv("NN_submit.csv", index=False)
print("\n### Final prediction from the Neural Network model")
print(df_nn_submit)

### Perform permutation importance
result = permutation_importance(model_nn, X, y, n_repeats=30, random_state=best_params_nn['Seed'])
importance_nn = result.importances_mean
feature_names_nn = X.columns
importance_df_nn = pd.DataFrame({'Feature': feature_names_nn, 'Importance': importance_nn})
importance_df_nn = importance_df_nn.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
colors = np.where(final_results_nn['testing_accuracy']< 0.83, 'red', \
                  np.where((final_results_nn['testing_accuracy'] >= 0.83) & (final_results_nn['testing_accuracy'] < 0.87),\
                            'orange', 'green'))
ax[0].scatter(final_results_nn.index, final_results_nn['testing_accuracy'], c=colors)
ax[0].set_title("Neural Network", fontsize=14)
ax[0].set_xlabel("Model index", fontsize=14)
ax[0].set_ylabel("Testing Accuracy", fontsize=14)
ax[0].set_ylim([0.8, 0.9])
ax[0].axhline(0.83, linestyle="--", color="k")
ax[0].axhline(0.87, linestyle="--", color="k")
ax[0].tick_params(axis='both', which='major', labelsize=14)

ax[1].barh(importance_df_nn['Feature'], importance_df_nn['Importance'])
ax[1].set_xlabel('Importance', fontsize=14)
ax[1].set_title('Feature Importances in Neural Network', fontsize=14)
ax[1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig("./NN.png")
plt.show()