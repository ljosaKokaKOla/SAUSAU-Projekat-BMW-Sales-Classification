from data_load_analysis import load_excel_data, null_statistic, anomaly_detection_num_values, onehot_encode,encode_target_label
#from data_visualize import scatter_visualize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
#Pocetak,ucitavanje podataka
def main():
    file_path = r"F:\SAUSAU_projekat\BMW_Car_Sales_Classification.csv"

    df_original = load_excel_data(file_path)

    if df_original is None:
        return
    #Provera za Null vrednosti,ostavljeno radi prikaza funkcionalnosti
    null_statistic(df_original)
    #Pocetne obrade Dataframea za dalje koriscenje
    df_clean_num = anomaly_detection_num_values(df_original)
    #df_ready = anomaly_detection_str_values(df_clean_num)
    df_encoded = onehot_encode(df_clean_num)
    df_encoded, target_encoder = encode_target_label(df_encoded)
    #Enkodiranje svih string vrednosti osim target
    X = df_encoded.drop('Sales_Classification', axis = 1)
    y = df_encoded['Sales_Classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 42)

    #plot_correlation_matrix(df_original, title= 'Korelacija originalnog skupa')

    correlation_matrix = df_original.select_dtypes(include = ['float64','int64']).corr()

    plt.figure(figsize = (20,18))
    sbn.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=.5)
    plt.title('Korelacija originalnog skupa',fontsize = 20)
    plt.tight_layout()
    plt.savefig("Korelacija_originalnog.png")
    plt.show(block=True)

    correlation_matrix_encoded = df_encoded.corr()

    plt.figure(figsize=(20, 18))
    sbn.heatmap(correlation_matrix_encoded, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Korelacija enkodiranog skupa', fontsize=20)
    plt.tight_layout()
    plt.savefig("Korelacija_encoded.png")
    plt.show(block=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = {}
    models = {}
    performance_metrics = {}

    print("\n------------------- Optimizacija modela: Logistička regresija -------------------")

    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # podržava l1 i l2
    }

    grid_search_lr = GridSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_grid=param_grid_lr,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search_lr.fit(X_train_scaled, y_train)

    best_lr = grid_search_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test_scaled)

    models['Logistička regresija'] = best_lr
    predictions['Logistička regresija'] = y_pred_lr

    print(f"Najbolji parametri: {grid_search_lr.best_params_}")
    print(f"Tačnost: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Preciznost (macro): {precision_score(y_test, y_pred_lr, average='macro'):.4f}")
    print(f"Odziv (macro): {recall_score(y_test, y_pred_lr, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred_lr, average='macro'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))

    print("\n------------------- Optimizacija modela: Decision Tree -------------------")

    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search_dt = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid_dt,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search_dt.fit(X_train, y_train)

    best_dt = grid_search_dt.best_estimator_
    y_pred_dt = best_dt.predict(X_test)

    models['Decision Tree'] = best_dt
    predictions['Decision Tree'] = y_pred_dt

    print(f"Najbolji parametri: {grid_search_dt.best_params_}")
    print(f"Tačnost: {accuracy_score(y_test, y_pred_dt):.4f}")
    print(f"Preciznost (macro): {precision_score(y_test, y_pred_dt, average='macro'):.4f}")
    print(f"Odziv (macro): {recall_score(y_test, y_pred_dt, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred_dt, average='macro'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_dt))

    from sklearn.neighbors import KNeighborsClassifier

    print("\n------------------- Optimizacija modela: KNN -------------------")

    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search_knn = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid_knn,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search_knn.fit(X_train_scaled, y_train)

    best_knn = grid_search_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test_scaled)

    models['KNN'] = best_knn
    predictions['KNN'] = y_pred_knn

    print(f"Najbolji parametri: {grid_search_knn.best_params_}")
    print(f"Tačnost: {accuracy_score(y_test, y_pred_knn):.4f}")
    print(f"Preciznost (macro): {precision_score(y_test, y_pred_knn, average='macro'):.4f}")
    print(f"Odziv (macro): {recall_score(y_test, y_pred_knn, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred_knn, average='macro'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_knn))

    print("\n------------------- Evaluacija modela: Naive Bayes -------------------")

    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    y_pred_nb = nb_model.predict(X_test_scaled)

    models['Naive Bayes'] = nb_model
    predictions['Naive Bayes'] = y_pred_nb

    print(f"Tačnost: {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"Preciznost (macro): {precision_score(y_test, y_pred_nb, average='macro'):.4f}")
    print(f"Odziv (macro): {recall_score(y_test, y_pred_nb, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred_nb, average='macro'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_nb))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nb))

    print("\n------------------- Rezultati i poređenje modela -------------------")

    # Računanje metrika za sve modele
    for model_name, y_pred in predictions.items():
        performance_metrics[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            'Recall': recall_score(y_test, y_pred, average='macro'),
            'F1-Score': f1_score(y_test, y_pred, average='macro')
        }

    # Prikaz tabele performansi
    print("\nTabela performansi:\n")
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("-" * 60)
    for model_name, metrics in performance_metrics.items():
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            model_name,
            metrics['Accuracy'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1-Score']
        ))

    # Prikaz matrica konfuzije
    for model_name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_)
        plt.xlabel('Predviđena klasa')
        plt.ylabel('Stvarna klasa')
        plt.title(f'Matrica konfuzije za {model_name}')
        plt.tight_layout()
        plt.show()

    X_reduced = X.drop(columns=['Region'], errors='ignore')

    scaler_reduced = StandardScaler()
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
        X_reduced, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_red_scaled = scaler_reduced.fit_transform(X_train_red)
    X_test_red_scaled = scaler_reduced.transform(X_test_red)

    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    grid_search_lr_red = GridSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_grid=param_grid_lr,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search_lr_red.fit(X_train_red_scaled, y_train_red)
    best_lr_red = grid_search_lr_red.best_estimator_
    y_pred_lr_red = best_lr_red.predict(X_test_red_scaled)

    print("\n--- Logistička regresija bez 'Region' ---")
    print(f"Najbolji parametri: {grid_search_lr_red.best_params_}")
    print(f"Tačnost: {accuracy_score(y_test_red, y_pred_lr_red):.4f}")
    print(f"Preciznost (macro): {precision_score(y_test_red, y_pred_lr_red, average='macro'):.4f}")
    print(f"Odziv (macro): {recall_score(y_test_red, y_pred_lr_red, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test_red, y_pred_lr_red, average='macro'):.4f}")

    cm_red = confusion_matrix(y_test_red, y_pred_lr_red)
    plt.figure(figsize=(6, 5))
    sbn.heatmap(cm_red, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    plt.xlabel('Predviđena klasa')
    plt.ylabel('Stvarna klasa')
    plt.title("Matrica konfuzije (LR bez 'Region')")
    plt.tight_layout()
    plt.show()

    df_metrics = pd.DataFrame.from_dict(performance_metrics, orient='index')

    fig, ax = plt.subplots(figsize=(10, 6))
    df_metrics.plot(kind='bar', ax=ax)

    plt.title('Performanse klasifikacionih modela', fontsize=14)
    plt.ylabel('Vrednost metrike')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()


#if df_original is not None:
    #print(df_original.head())
#("----------------------------------------------")
#print(df_original.info())
#print(df_original.describe())
#print("----------------------------------------------")

#if df_original is not None:
   # null_statistic(df_original)

#df_clean_num_anom = anomaly_detection_num_values(df_original)
#print("----------------------------------------------")
#df_ready = anomaly_detection_str_values(df_clean_num_anom)
#print("----------------------------------------------")
#df_encoded = onehot_encode(df_ready)

#scatter_visualize(df_original,'Engine_Size_L')
#scatter_visualize(df_clean_num_anom,'Engine_Size_L')

if __name__ == '__main__':
    main()
