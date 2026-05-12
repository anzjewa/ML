"""
ЛАБОРАТОРНАЯ РАБОТА
Линейные модели, SVM и деревья решений

Дисциплина: Технологии разведочного анализа и обработки данных
Набор данных: успеваемость.csv
Задача: Классификация (предсказание сдачи экзамена)
"""

# ============================================================
# 1. ИМПОРТ БИБЛИОТЕК
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# Настройка визуализации
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("="*80)
print("ЛАБОРАТОРНАЯ РАБОТА: Линейные модели, SVM и деревья решений")
print("="*80)

# ============================================================
# 2. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ============================================================

print("\n1. ЗАГРУЗКА ДАННЫХ")
print("-"*50)

# Загрузка данных из файла
df = pd.read_csv('успеваемость.csv')
print(f"Размер датасета: {df.shape}")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nИнформация о данных:")
print(df.info())
print(f"\nСтатистика пропусков:")
print(df.isnull().sum())

# ============================================================
# 3. ПРЕДОБРАБОТКА ДАННЫХ
# ============================================================

print("\n" + "="*80)
print("2. ПРЕДОБРАБОТКА ДАННЫХ")
print("="*80)

df_clean = df.copy()

# Удаляем столбец с ID студента (не нужен для обучения)
if 'студент_id' in df_clean.columns:
    df_clean = df_clean.drop('студент_id', axis=1)
    print("✓ Удален столбец 'студент_id'")

# Проверка и заполнение пропусков
print("\nПроверка пропусков:")
for col in df_clean.columns:
    null_count = df_clean[col].isnull().sum()
    if null_count > 0:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"  {col}: заполнено {null_count} пропусков медианой")
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            print(f"  {col}: заполнено {null_count} пропусков модой")
    else:
        print(f"  {col}: пропусков нет")

# Кодирование категориальных признаков (если есть)
print("\nКодирование категориальных признаков:")
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    print(f"  {col}: закодирован (уникальных значений: {len(le.classes_)})")

print(f"\nИтоговый размер данных: {df_clean.shape}")

# ============================================================
# 4. РАЗДЕЛЕНИЕ ПРИЗНАКОВ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ============================================================

print("\n" + "="*80)
print("3. РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
print("="*80)

# Определяем признаки (X) и целевую переменную (y)
X = df_clean.drop('сдал_экзамен', axis=1)
y = df_clean['сдал_экзамен']

print(f"Признаки (X): {list(X.columns)}")
print(f"Целевая переменная (y): сдал_экзамен")
print(f"\nРаспределение целевой переменной:")
print(f"  Сдало экзамен: {sum(y == 1)} студентов ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"  Не сдало экзамен: {sum(y == 0)} студентов ({sum(y == 0)/len(y)*100:.1f}%)")

# Масштабирование признаков (для логистической регрессии и SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]} образцов")

# ============================================================
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ И ОЦЕНКА КАЧЕСТВА
# ============================================================

print("\n" + "="*80)
print("4. ОБУЧЕНИЕ МОДЕЛЕЙ И ОЦЕНКА КАЧЕСТВА")
print("="*80)

# Метрики для оценки:
# - Accuracy (точность): доля правильных ответов
# - F1-score: гармоническое среднее точности и полноты (лучше для несбалансированных классов)

results = {}

# ------------------------------------------------------------
# МОДЕЛЬ 1: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ------------------------------------------------------------

print("\n" + "-"*50)
print("МОДЕЛЬ 1: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("-"*50)

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

results['Логистическая регрессия'] = {
    'Accuracy': accuracy_logreg,
    'F1-score': f1_logreg
}

print(f"Accuracy:  {accuracy_logreg:.4f}")
print(f"F1-score:  {f1_logreg:.4f}")
print(f"\nПодробный отчет классификации:")
print(classification_report(y_test, y_pred_logreg, target_names=['Не сдал', 'Сдал']))

# Коэффициенты логистической регрессии (важность признаков)
feature_importance_logreg = pd.DataFrame({
    'Признак': X.columns,
    'Коэффициент': logreg.coef_[0]
}).sort_values('Коэффициент', ascending=False)

print(f"\nВажность признаков (коэффициенты логистической регрессии):")
for idx, row in feature_importance_logreg.iterrows():
    print(f"  {row['Признак']}: {row['Коэффициент']:.4f}")

# ------------------------------------------------------------
# МОДЕЛЬ 2: SVM (МЕТОД ОПОРНЫХ ВЕКТОРОВ)
# ------------------------------------------------------------

print("\n" + "-"*50)
print("МОДЕЛЬ 2: SVM (МЕТОД ОПОРНЫХ ВЕКТОРОВ)")
print("-"*50)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

results['SVM'] = {
    'Accuracy': accuracy_svm,
    'F1-score': f1_svm
}

print(f"Accuracy:  {accuracy_svm:.4f}")
print(f"F1-score:  {f1_svm:.4f}")
print(f"\nПодробный отчет классификации:")
print(classification_report(y_test, y_pred_svm, target_names=['Не сдал', 'Сдал']))

# ------------------------------------------------------------
# МОДЕЛЬ 3: ДЕРЕВО РЕШЕНИЙ
# ------------------------------------------------------------

print("\n" + "-"*50)
print("МОДЕЛЬ 3: ДЕРЕВО РЕШЕНИЙ")
print("-"*50)

tree = DecisionTreeClassifier(random_state=42, max_depth=4)  # ограничиваем глубину для наглядности
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

accuracy_tree = accuracy_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

results['Дерево решений'] = {
    'Accuracy': accuracy_tree,
    'F1-score': f1_tree
}

print(f"Accuracy:  {accuracy_tree:.4f}")
print(f"F1-score:  {f1_tree:.4f}")
print(f"\nПодробный отчет классификации:")
print(classification_report(y_test, y_pred_tree, target_names=['Не сдал', 'Сдал']))

# ============================================================
# 6. СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================

print("\n" + "="*80)
print("5. СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ")
print("="*80)

# Создаем DataFrame для сравнения
comparison_df = pd.DataFrame(results).T
print("\nСравнение метрик качества моделей:")
print(comparison_df.round(4))

print("\n" + "-"*50)
print("ВЫВОДЫ ПО СРАВНЕНИЮ МОДЕЛЕЙ:")
print("-"*50)

# Определяем лучшую модель по F1-score
best_model = comparison_df['F1-score'].idxmax()
best_score = comparison_df['F1-score'].max()

print(f"\n1. Лучшая модель по F1-score: {best_model} (F1 = {best_score:.4f})")

# Сравнение всех моделей
for model in results.keys():
    if results[model]['Accuracy'] >= 0.8:
        print(f"2. {model}: качество выше 80% - модель пригодна для использования")
    else:
        print(f"2. {model}: качество ниже 80% - требуется дополнительная настройка")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ ВАЖНОСТИ ПРИЗНАКОВ В ДЕРЕВЕ РЕШЕНИЙ
# ============================================================

print("\n" + "="*80)
print("6. ВАЖНОСТЬ ПРИЗНАКОВ В ДЕРЕВЕ РЕШЕНИЙ")
print("="*80)

# Получаем важность признаков
feature_importance = tree.feature_importances_

# Создаем DataFrame для визуализации
importance_df = pd.DataFrame({
    'Признак': X.columns,
    'Важность': feature_importance
}).sort_values('Важность', ascending=False)

print("\nВажность признаков (Gini importance):")
for idx, row in importance_df.iterrows():
    print(f"  {row['Признак']}: {row['Важность']:.4f} ({row['Важность']*100:.1f}%)")

# Построение графика важности признаков
plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))
bars = plt.barh(importance_df['Признак'], importance_df['Важность'], color=colors, edgecolor='black')
plt.xlabel('Важность признака', fontsize=12)
plt.ylabel('Признаки', fontsize=12)
plt.title('Важность признаков в модели Дерево решений', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Инвертируем ось для отображения самого важного сверху

# Добавляем значения на столбцы
for bar, val in zip(bars, importance_df['Важность']):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.show()

print("\nИнтерпретация важности признаков:")
print("  • Чем выше значение важности, тем больше вклад признака в принятие решений")
print("  • Сумма всех важностей равна 1")
print(f"  • Наиболее важный признак: {importance_df.iloc[0]['Признак']} ({importance_df.iloc[0]['Важность']*100:.1f}%)")

# ============================================================
# 8. ВИЗУАЛИЗАЦИЯ ДЕРЕВА РЕШЕНИЙ
# ============================================================

print("\n" + "="*80)
print("7. ВИЗУАЛИЗАЦИЯ ДЕРЕВА РЕШЕНИЙ")
print("="*80)

# Вариант 1: Графическое отображение дерева
plt.figure(figsize=(16, 10))
plot_tree(
    tree, 
    feature_names=X.columns, 
    class_names=['Не сдал', 'Сдал'],
    filled=True, 
    rounded=True,
    fontsize=10,
    impurity=False
)
plt.title('Дерево решений (визуализация)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Вариант 2: Текстовое представление правил дерева
print("\nПРАВИЛА ДЕРЕВА РЕШЕНИЙ (текстовое представление):")
print("-"*50)
tree_rules = export_text(tree, feature_names=list(X.columns), show_weights=True)
print(tree_rules)

# ============================================================
# 9. ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ДЛЯ АНАЛИЗА
# ============================================================

print("\n" + "="*80)
print("8. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("="*80)

# 9.1 Матрицы ошибок для всех моделей
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    (logreg, y_pred_logreg, 'Логистическая регрессия', 'Blues'),
    (svm, y_pred_svm, 'SVM', 'Greens'),
    (tree, y_pred_tree, 'Дерево решений', 'Oranges')
]

for idx, (model, y_pred, title, cmap) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx])
    axes[idx].set_title(f'{title}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}', fontsize=12)
    axes[idx].set_xlabel('Предсказано')
    axes[idx].set_ylabel('Фактически')
    axes[idx].set_xticklabels(['Не сдал', 'Сдал'])
    axes[idx].set_yticklabels(['Не сдал', 'Сдал'])

plt.suptitle('Матрицы ошибок для всех моделей', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 9.2 Сравнение моделей на графике
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'F1-score']
x = np.arange(len(metrics))
width = 0.25

for i, (model_name, scores) in enumerate(results.items()):
    values = [scores['Accuracy'], scores['F1-score']]
    pos = x + i * width
    bars = plt.bar(pos, values, width, label=model_name, edgecolor='black')
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.xlabel('Метрики качества', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.title('Сравнение качества моделей классификации', fontsize=14, fontweight='bold')
plt.xticks(x + width, metrics)
plt.legend()
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ============================================================
# 10. ИТОГОВЫЕ ВЫВОДЫ
# ============================================================

print("\n" + "="*80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("="*80)

print("""
РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ ЛАБОРАТОРНОЙ РАБОТЫ:

1. ЗАДАЧА КЛАССИФИКАЦИИ:
   • Целевая переменная: 'сдал_экзамен' (1 - сдал, 0 - не сдал)
   • Количество признаков: 5
   • Размер выборки: 20 студентов

2. КАЧЕСТВО МОДЕЛЕЙ (по F1-score):
""")
for model_name, scores in results.items():
    print(f"   • {model_name}: {scores['F1-score']:.4f}")

print("""
3. ВЫВОДЫ ПО МОДЕЛЯМ:
   • Логистическая регрессия: простая и интерпретируемая модель,
     позволяет оценить вклад каждого признака через коэффициенты
   
   • SVM: эффективна при сложных границах между классами,
     чувствительна к масштабированию признаков
   
   • Дерево решений: наиболее интерпретируемая модель,
     позволяет визуализировать процесс принятия решений

4. НАИБОЛЕЕ ВАЖНЫЕ ПРИЗНАКИ (по версии дерева решений):
""")
for idx, row in importance_df.iterrows():
    print(f"   • {row['Признак']}: {row['Важность']*100:.1f}%")

print("""
5. РЕКОМЕНДАЦИИ:
   • Для использования в производственной среде рекомендуется модель,
     показавшую наилучшее значение F1-score
   • Для интерпретируемости решений лучше использовать дерево решений
   • Для простоты и скорости - логистическую регрессию
""")

print("\n" + "="*80)
print("ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА")
print("="*80)