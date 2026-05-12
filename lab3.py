import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Загрузка данных из файла успеваемость.csv
df = pd.read_csv('успеваемость.csv')
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {list(df.columns)}\n")

# Создаем копию для обработки
df_clean = df.copy()

# Удаляем ненужные колонки (если есть)
columns_to_drop = ['студент_id']  # ID студента не нужен для обучения
df_clean = df_clean.drop([col for col in columns_to_drop if col in df_clean.columns], axis=1)

# Заполнение пропусков (если есть)
print("Проверка пропусков до обработки:")
print(df_clean.isnull().sum())
print("\n")

# Заполняем пропуски (если есть)
for col in df_clean.columns:
    if df_clean[col].dtype in ['float64', 'int64']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")

print("После обработки пропусков:")
print(df_clean.isnull().sum())
print()

# Кодирование категориальных признаков (если есть)
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col])
    print(f"Закодирован признак: {col}")

print()

# Разделяем на признаки и целевую переменную
X = df_clean.drop('сдал_экзамен', axis=1)
y = df_clean['сдал_экзамен']

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Обучающая выборка: {X_train.shape[0]} образцов, Тестовая: {X_test.shape[0]} образцов\n")
print("Признаки:", list(X.columns))
print("\n" + "="*60)

# Исходная модель KNN с K=7
knn_initial = KNeighborsClassifier(n_neighbors=7)
knn_initial.fit(X_train, y_train)
y_pred_initial = knn_initial.predict(X_test)

print("\nИСХОДНАЯ МОДЕЛЬ (K=7)")
print("-"*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_initial):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_initial):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_initial):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_initial):.4f}")

# Подбор гиперпараметров
param_grid = {'n_neighbors': np.arange(1, 51, 2)}  # 1, 3, 5, ..., 49

# Стратегия 1: KFold (5 фолдов)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_kf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kf, scoring='f1', n_jobs=-1)
grid_kf.fit(X_train, y_train)

# Стратегия 2: ShuffleSplit (10 сплитов)
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
grid_ss = GridSearchCV(KNeighborsClassifier(), param_grid, cv=ss, scoring='f1', n_jobs=-1)
grid_ss.fit(X_train, y_train)

# RandomizedSearchCV
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ПОДБОРА ГИПЕРПАРАМЕТРОВ")
print("-"*60)
print(f"GridSearchCV (KFold):        K = {grid_kf.best_params_['n_neighbors']}, F1 = {grid_kf.best_score_:.4f}")
print(f"GridSearchCV (ShuffleSplit): K = {grid_ss.best_params_['n_neighbors']}, F1 = {grid_ss.best_score_:.4f}")
print(f"RandomizedSearchCV:          K = {random_search.best_params_['n_neighbors']}, F1 = {random_search.best_score_:.4f}")

# Оптимальная модель
best_k = grid_kf.best_params_['n_neighbors']
knn_optimal = KNeighborsClassifier(n_neighbors=best_k)
knn_optimal.fit(X_train, y_train)
y_pred_optimal = knn_optimal.predict(X_test)

print("\n" + "="*60)
print(f"ОПТИМАЛЬНАЯ МОДЕЛЬ (K={best_k})")
print("-"*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_optimal):.4f}")

# Сравнение моделей
print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ".center(60))
print("="*60)
print(f"{'Метрика':<12} | {'Исходная (K=7)':<18} | {'Оптимальная (K=' + str(best_k) + ')':<20}")
print("-"*60)
print(f"{'Accuracy':<12} | {accuracy_score(y_test, y_pred_initial):.4f}{'':<14} | {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"{'Precision':<12} | {precision_score(y_test, y_pred_initial):.4f}{'':<14} | {precision_score(y_test, y_pred_optimal):.4f}")
print(f"{'Recall':<12} | {recall_score(y_test, y_pred_initial):.4f}{'':<14} | {recall_score(y_test, y_pred_optimal):.4f}")
print(f"{'F1-score':<12} | {f1_score(y_test, y_pred_initial):.4f}{'':<14} | {f1_score(y_test, y_pred_optimal):.4f}")

# График зависимости качества от K
results = pd.DataFrame(grid_kf.cv_results_)
plt.figure(figsize=(10, 5))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], 'o-', color='navy', label='F1 на CV')
plt.fill_between(results['param_n_neighbors'],
                 results['mean_test_score'] - results['std_test_score'],
                 results['mean_test_score'] + results['std_test_score'], alpha=0.2, color='navy')
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Оптимальное K = {best_k}')
plt.xlabel('Количество соседей (K)', fontsize=12)
plt.ylabel('F1-score', fontsize=12)
plt.title('Зависимость качества модели KNN от гиперпараметра K', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Матрицы ошибок
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

cm_initial = confusion_matrix(y_test, y_pred_initial)
sns.heatmap(cm_initial, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Матрица ошибок (K=7)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Предсказано', fontsize=10)
axes[0].set_ylabel('Фактически', fontsize=10)

cm_optimal = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Матрица ошибок (K={best_k})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Предсказано', fontsize=10)
axes[1].set_ylabel('Фактически', fontsize=10)

plt.tight_layout()
plt.show()

# Дополнительный анализ: важность признаков (для KNN можно посмотреть веса)
print("\n" + "="*60)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("="*60)
print(f"\nЛучшее значение K: {best_k}")
print(f"Лучший F1-score на кросс-валидации: {grid_kf.best_score_:.4f}")
print(f"Улучшение F1-score: {(f1_score(y_test, y_pred_optimal) - f1_score(y_test, y_pred_initial)):.4f}")

# Анализ распределения классов
print("\nРаспределение целевой переменной:")
print(f"  Сдало экзамен: {sum(y == 1)} студентов ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"  Не сдало экзамен: {sum(y == 0)} студентов ({sum(y == 0)/len(y)*100:.1f}%)")