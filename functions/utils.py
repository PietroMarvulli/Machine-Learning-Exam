import copy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
output_folder = r'.\functions\images'
os.makedirs(output_folder, exist_ok=True)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

def tsne2D(data):
    classes = data["Outcome"]
    data = data.values[:,:7]
    tsne = TSNE(n_components=2, random_state=1)
    tsne_results = tsne.fit_transform(data)

    data_scatter = pd.DataFrame(np.column_stack((tsne_results, classes)),columns=["x", "y", "label"])
    fig = plt.figure()
    sns.scatterplot(data=data_scatter, x="x", y="y", hue="label",palette="deep",legend="full", s=30)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # ax.tick_params(labelsize=70, length=0)
    # cax = heatmap.collections[0].colorbar.ax
    # cax.tick_params(labelsize=80)
    plt.title("TSNE", fontsize=20)
    plt.show()
    plt.savefig("./functions/images/TSNE.png")
    print(0)




def correlation_matrix(data):
    data = copy.deepcopy(data)
    data.drop(columns = "Outcome", axis = 1, inplace = True)
    mat = data.corr()
    print("Save Correlation Matrix")

    fig, ax = plt.subplots(figsize=(100, 100))
    heatmap = sns.heatmap(mat, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=True, yticklabels=True,annot_kws={'size': 80}, square=True)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(labelsize=70, length=0)
    cax = heatmap.collections[0].colorbar.ax
    cax.tick_params(labelsize=80)
    plt.title("Correlation Matrix", fontsize=100)
    plt.savefig("./functions/images/Correlation_Matrix.png")

    correlation_threshold = 0.70
    high_correlation_pairs = []
    for i in range(len(mat.columns)):
        for j in range(i + 1, len(mat.columns)):
            if abs(mat.iloc[i, j]) > correlation_threshold:
                v1 = mat.columns[i]
                v2 = mat.columns[j]
                correlation_value = mat.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    drop_list = []
    if not drop_list:
        print("Features are not correlated each others")
    else:
        for pair in high_correlation_pairs:
            print(f"Coppia: {pair[0]} - {pair[1]}, Correlazione: {pair[2]}")
            count_first = len(data[pair[0]].unique())
            count_second = len(data[pair[1]].unique())
            if count_first >= count_second:
                drop_list.append(pair[0])
                print(f"Della Coppia: {pair[0]} - {pair[1]}, con Correlazione: {pair[2]} elimino la Feature {pair[0]}")
            else:
                drop_list.append(pair[1])
                print(f"Della Coppia: {pair[0]} - {pair[1]}, con Correlazione: {pair[2]} elimino la Feature {pair[1]}")



def save_features_plot(dataframe, save_path, pathology):
    sns.set(style="darkgrid")
    for feature in dataframe.columns[:-1]:
        if feature == 'Pregnancies':
            plt.figure(figsize=(15, 10))
            plt.tight_layout()
            sns.countplot(x=feature, data=dataframe, orient="h", color='steelblue')
            plt.xlabel('Pregnancies', fontsize=20)
            plt.ylabel('Patients Number', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            save_file = f"{save_path}/{feature}_{pathology}.png"
            plt.savefig(save_file)
        else:
            plt.figure(figsize=(15, 10))
            plt.tight_layout()
            sns.boxplot(x=feature, y=dataframe.columns[-1], data=dataframe, orient="h")
            plt.xlabel(feature, fontsize=20)
            plt.ylabel(dataframe.columns[-1], fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylim(-0.5, 1.5)
            save_file = f"{save_path}/{feature}_{pathology}.png"
            plt.savefig(save_file)


def plot_conf_matrix(y_true, y_pred, model_name="model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    cm_reordered = np.array([
        [cm[1, 1], cm[1, 0]],
        [cm[0, 1], cm[0, 0]]
    ])

    color_matrix = np.array([
        [1, 0],
        [0, 1]
    ])

    cmap = ListedColormap(["red", "green"])

    plt.figure(figsize=(6, 6))
    plt.pcolor(color_matrix, cmap=cmap, edgecolors='black', linewidths=2)

    labels = np.array([
        [f"{cm_reordered[0, 0]}\n(True Positive)", f"{cm_reordered[0, 1]}\n(False Negative)"],
        [f"{cm_reordered[1, 0]}\n(False Positive)", f"{cm_reordered[1, 1]}\n(True Negative)"]
    ])

    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.5, labels[i, j],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=16, color='white', weight='bold')

    plt.xticks([0.5, 1.5], ['True', 'False'], fontsize=12)
    plt.yticks([0.5, 1.5], ['True', 'False'], fontsize=12)
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Actual Values", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, f"{model_name}_custom_conf_matrix.png"))
    plt.show()

def plot_roc_curve(y_true, y_scores, model_name="model", save_path=None):
    RocCurveDisplay.from_predictions(y_true, y_scores)
    plt.title("ROC Curve")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{model_name}_roc.png"))

def feature_processing(dataframe):
    dataframe = dataframe.copy()

    for col in dataframe.columns:
        if col not in ['Pregnancies', 'Outcome']:
            dataframe[col] = dataframe[col].replace(0, np.nan)
    dataframe = dataframe.dropna()

    print(f"Numero pazienti rimanenti: {len(dataframe)}")
    print(f"Numero diabetici: {dataframe['Outcome'].sum()}")

    features = dataframe.drop(columns=['Outcome'])
    labels = dataframe['Outcome']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    df_scaled['Outcome'] = labels.reset_index(drop=True)
    return df_scaled


def univariate_features(dataframe, fpr=True, fdr=False):
    X = dataframe.drop(columns=['Outcome'])
    y = dataframe['Outcome']

    if fpr:
        selector = SelectFpr(score_func=f_classif, alpha=0.05)
    else:
        raise ValueError("Solo FPR supportato per ora")

    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Features selezionate: {list(selected_features)}")

    df_selected = pd.DataFrame(X_new, columns=selected_features)
    df_selected['Outcome'] = y.reset_index(drop=True)
    return df_selected


def performance_evaluation(y_train_true, y_train_pred, y_train_proba,
                           y_test_true, y_test_pred, y_test_proba):
    # AUC
    auc_train = roc_auc_score(y_train_true, y_train_proba)
    auc_test = roc_auc_score(y_test_true, y_test_proba)

    # Precision, Recall, F1-score
    precision_train = precision_score(y_train_true, y_train_pred)
    precision_test = precision_score(y_test_true, y_test_pred)

    recall_train = recall_score(y_train_true, y_train_pred)
    recall_test = recall_score(y_test_true, y_test_pred)

    f1_train = f1_score(y_train_true, y_train_pred)
    f1_test = f1_score(y_test_true, y_test_pred)

    metrics_data = {
        "Metric": ["AUC", "Precision", "Recall", "F1-score"],
        "Train Set": [auc_train, precision_train, recall_train, f1_train],
        "Test Set": [auc_test, precision_test, recall_test, f1_test]
    }
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df)
    return metrics_df

def logistic_regression(train, test, save_fig=False):
    X_train = train.drop(columns='Outcome')
    y_train = train['Outcome']
    X_test = test.drop(columns='Outcome')
    y_test = test['Outcome']

    param_grid = {
        'C': [0.01, 0.1, 1.0],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }

    clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    print("Logistic Regression")
    metrics_df = performance_evaluation(y_train, y_train_pred, y_train_proba,
                                        y_test, y_test_pred, y_test_proba)

    if save_fig:
        plot_roc_curve(y_test, y_test_proba, model_name="LR", save_path=output_folder)
        plot_conf_matrix(y_test, y_test_pred, model_name="LR", save_path=output_folder)

    return metrics_df



def decision_tree(train, test, save_fig=False):
    X_train = train.drop(columns='Outcome')
    y_train = train['Outcome']
    X_test = test.drop(columns='Outcome')
    y_test = test['Outcome']

    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 3, 5]
    }

    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    print('Decision Tree')
    metrics_df = performance_evaluation(y_train, y_train_pred, y_train_proba,
                                        y_test, y_test_pred, y_test_proba)

    if save_fig:
        plot_roc_curve(y_test, y_test_proba, model_name="DT", save_path=output_folder)
        plot_conf_matrix(y_test, y_test_pred, model_name="DT", save_path=output_folder)

    return metrics_df




def support_vector_machine(train, test, save_fig=False):
    X_train = train.drop(columns='Outcome')
    y_train = train['Outcome']
    X_test = test.drop(columns='Outcome')
    y_test = test['Outcome']

    base_svm = LinearSVC(dual=False, random_state=42, max_iter=10000)
    svm = CalibratedClassifierCV(estimator = base_svm, method='sigmoid', cv=5)

    param_grid = {
        'estimator__C': [0.01, 0.1, 1.0]
    }

    clf = GridSearchCV(svm, param_grid, cv=5)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    print('Support Vector Machines')
    metrics_df = performance_evaluation(y_train, y_train_pred, y_train_proba,
                                        y_test, y_test_pred, y_test_proba)

    if save_fig:
        plot_roc_curve(y_test, y_test_proba, model_name="SVM", save_path=output_folder)
        plot_conf_matrix(y_test, y_test_pred, model_name="SVM", save_path=output_folder)

    return metrics_df
