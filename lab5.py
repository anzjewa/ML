import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

print("="*70)
print("ЛАБОРАТОРНАЯ РАБОТА: АНСАМБЛЕВЫЕ МЕТОДЫ")
print("Random Forest, Extra Trees, AdaBoost, Gradient Boosting")
print("="*70)

# 1. Загрузка данных
print("\n1. ЗАГРУЗКА ДАННЫХ")
print("-"*50)
df = pd.read_csv("успеваемость.csv")
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {list(df.columns)}")
print("\nПервые 5 строк:")
print(df.head())

# 2. Предобработка данных
print("\n2. ПРЕДОБРАБОТКА ДАННЫХ")
print("-"*50)

# Удаляем ненужные признаки (студент_id не нужен для обучения)
if 'студент_id' in df.columns:
    df = df.drop("студент_id", axis=1)
    print("✓ Удалена колонка 'студент_id'")

# Заполняем числовые пропуски (если есть)
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
        print(f"✓ Заполнены пропуски в колонке '{col}' медианой")

# Заполняем категориальные пропуски (если есть)
for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
        print(f"✓ Заполнены пропуски в колонке '{col}' модой")

# Проверка пропусков
print("\nПропуски после обработки:")
print(df.isnull().sum())

# Кодирование категориальных признаков
print("\n3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("-"*50)
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=["object"]).columns

if len(categorical_cols) > 0:
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"✓ Закодирована колонка '{col}'")
else:
    print("Категориальные признаки отсутствуют")

print("\nТипы данных после обработки:")
print(df.dtypes)

# 3. Разделение данных
print("\n4. РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
print("-"*50)

X = df.drop("сдал_экзамен", axis=1)
y = df["сдал_экзамен"]

print(f"Признаки (X): {list(X.columns)}")
print(f"Целевая переменная (y): сдал_экзамен")
print(f"\nРаспределение целевой переменной:")
print(f"  Сдало экзамен: {sum(y == 1)} студентов ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"  Не сдало экзамен: {sum(y == 0)} студентов ({sum(y == 0)/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]} образцов")

# 4. Обучение моделей
print("\n5. ОБУЧЕНИЕ АНСАМБЛЕВЫХ МОДЕЛЕЙ")
print("="*70)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 5. Оценка моделей
print("\n6. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("="*70)

results = {}
detailed_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results[name] = acc
    detailed_results[name] = {
        'Accuracy': acc,
        'F1-score': f1,
        'Precision': precision,
        'Recall': recall
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy  = {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-score  = {f1:.4f}")
    print(f"  Precision = {precision:.4f}")
    print(f"  Recall    = {recall:.4f}")
    print("-" * 40)

# 6. Сравнение моделей
print("\n7. СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*70)

print("\nСводная таблица результатов (Accuracy):")
print(f"\n{'Модель':<20} {'Accuracy':<12} {'Рейтинг':<10}")
print("-" * 45)

# Сортируем по accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i, (name, score) in enumerate(sorted_results, 1):
    print(f"{name:<20} {score:<12.4f} {i}-е место")

# Определяем лучшую модель
best_model = max(results, key=results.get)
best_score = results[best_model]

print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model}")
print(f"   Accuracy = {best_score:.4f} ({best_score*100:.2f}%)")

# 7. Дополнительный анализ: важность признаков
print("\n8. ВАЖНОСТЬ ПРИЗНАКОВ")
print("="*70)

# Используем Random Forest для анализа важности признаков
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_names = X.columns
importances = rf_model.feature_importances_

# Сортировка по важности
importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': importances
}).sort_values('Важность', ascending=False)

print("\nВажность признаков (Random Forest):")
for idx, row in importance_df.iterrows():
    print(f"  {row['Признак']}: {row['Важность']:.4f} ({row['Важность']*100:.1f}%)")

# 8. Анализ работы лучшей модели
print("\n9. АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ")
print("="*70)

best_model_obj = models[best_model]
y_pred_best = best_model_obj.predict(X_test)

print(f"\nМодель: {best_model}")
print(f"Количество деревьев/оценщиков: {best_model_obj.n_estimators if hasattr(best_model_obj, 'n_estimators') else 'N/A'}")

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred_best)
print(f"\nМатрица ошибок:")
print(f"""
          Предсказано
          Не сдал  Сдал
Фактически:
Не сдал     {cm[0,0]}      {cm[0,1]}
Сдал        {cm[1,0]}      {cm[1,1]}
""")

print(f"\nПодробный отчет классификации:")
print(classification_report(y_test, y_pred_best, target_names=['Не сдал', 'Сдал']))

# 9. Визуализация сравнения моделей
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
models_names = list(results.keys())
scores = list(results.values())

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = plt.bar(models_names, scores, color=colors, edgecolor='black')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Модели', fontsize=12)
plt.title('Сравнение точности (Accuracy) ансамблевых методов', fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)

# Добавляем значения на столбцы
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.3f}', ha='center', va='bottom', fontsize=11)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 10. График важности признаков
plt.figure(figsize=(10, 6))
colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(importance_df)))
bars = plt.barh(importance_df['Признак'], importance_df['Важность'], color=colors, edgecolor='black')
plt.xlabel('Важность признака', fontsize=12)
plt.ylabel('Признаки', fontsize=12)
plt.title('Важность признаков (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for bar, val in zip(bars, importance_df['Важность']):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.show()

# 11. ИТОГОВЫЕ ВЫВОДЫ
print("\n" + "="*70)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("="*70)

print(f"""
1. ЗАДАЧА: Бинарная классификация (предсказание сдачи экзамена)

2. РАЗМЕР ВЫБОРКИ:
   • Всего студентов: {len(df)}
   • Обучающая выборка: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.0f}%)
   • Тестовая выборка: {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.0f}%)

3. РЕЗУЛЬТАТЫ МОДЕЛЕЙ (Accuracy):
""")
for name, score in sorted_results:
    print(f"   • {name}: {score:.4f}")

print(f"""
4. ЛУЧШАЯ МОДЕЛЬ: {best_model} (Accuracy = {best_score:.4f})

5. НАИБОЛЕЕ ВАЖНЫЕ ПРИЗНАКИ:
""")
for idx, row in importance_df.head(3).iterrows():
    print(f"   • {row['Признак']}: {row['Важность']*100:.1f}%")

print("""
6. ВЫВОДЫ ПО АНСАМБЛЕВЫМ МЕТОДАМ:
   • Random Forest: хорошо работает по умолчанию, устойчив к переобучению
   • Extra Trees: добавляет дополнительную случайность, может быть быстрее
   • AdaBoost: последовательно улучшает слабые классификаторы
   • Gradient Boosting: строит деревья последовательно, исправляя ошибки предыдущих

7. РЕКОМЕНДАЦИИ:
   • Для данного набора данных лучше всего показала себя модель {best_model}
   • Наиболее важными факторами успеваемости являются: {importance_df.iloc[0]['Признак']} и {importance_df.iloc[1]['Признак']}
   • Для улучшения качества можно попробовать настроить гиперпараметры
""")

print("\n" + "="*70)
print("="*70)
