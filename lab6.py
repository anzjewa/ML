import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Стекинг
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Многослойный персептрон (MLP)
from sklearn.neural_network import MLPClassifier

# Для МГУА (попытка импорта, но библиотека может быть не установлена)
try:
    # GMDH для Python от bauman-team
    from gmdh import Combi, Multi, MIA, RIA, split_data
    GMDH_AVAILABLE = True
    print("✓ Библиотека gmdh успешно загружена")
except ImportError:
    GMDH_AVAILABLE = False
    print("! Библиотека gmdh не установлена. МГУА модели не будут обучены.")
    print("  Для установки выполните: pip install gmdh")

print("="*80)
print("ЛАБОРАТОРНАЯ РАБОТА: АНСАМБЛИ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ. ЧАСТЬ 2")
print("Стекинг, Многослойный персептрон (MLP), МГУА")
print("="*80)

# ============================================================
# 2. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ============================================================

print("\n1. ЗАГРУЗКА ДАННЫХ")
print("-"*50)

df = pd.read_csv("успеваемость.csv")
print(f"Размер датасета: {df.shape}")
print(f"\nПервые 5 строк:")
print(df.head())

# ============================================================
# 3. ПРЕДОБРАБОТКА ДАННЫХ
# ============================================================

print("\n2. ПРЕДОБРАБОТКА ДАННЫХ")
print("-"*50)

df_clean = df.copy()

# Удаляем столбец с ID студента
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

# Кодирование категориальных признаков
print("\nКодирование категориальных признаков:")
categorical_cols = df_clean.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        df_clean[col] = le.fit_transform(df_clean[col])
        print(f"  {col}: закодирован")
else:
    print("  Категориальные признаки отсутствуют")

print(f"\nИтоговый размер данных: {df_clean.shape}")

# ============================================================
# 4. РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ============================================================

print("\n3. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-"*50)

X = df_clean.drop('сдал_экзамен', axis=1)
y = df_clean['сдал_экзамен']

print(f"Признаки (X): {list(X.columns)}")
print(f"Целевая переменная (y): сдал_экзамен")
print(f"\nРаспределение целевой переменной:")
print(f"  Сдало экзамен: {sum(y == 1)} студентов ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"  Не сдало экзамен: {sum(y == 0)} студентов ({sum(y == 0)/len(y)*100:.1f}%)")

# Масштабирование (для MLP и стекинга)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]} образцов")

# ============================================================
# 5. МОДЕЛЬ 1: СТЕКИНГ (STACKING CLASSIFIER)
# ============================================================

print("\n" + "="*80)
print("4. МОДЕЛЬ 1: СТЕКИНГ (Stacking Classifier)")
print("="*80)

# Базовые классификаторы для стекинга
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Мета-классификатор (логистическая регрессия)
meta_learner = LogisticRegression(random_state=42, max_iter=1000)

# Создание модели стекинга
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # 5-кратная кросс-валидация для обучения мета-модели
    stack_method='predict_proba'
)

# Обучение
stacking_model.fit(X_train, y_train)

# Предсказание
y_pred_stacking = stacking_model.predict(X_test)

# Оценка качества
acc_stacking = accuracy_score(y_test, y_pred_stacking)
f1_stacking = f1_score(y_test, y_pred_stacking)

print(f"Результаты модели Стекинг:")
print(f"  Accuracy = {acc_stacking:.4f} ({acc_stacking*100:.2f}%)")
print(f"  F1-score = {f1_stacking:.4f}")

print(f"\nБазовые модели в ансамбле:")
for name, model in base_learners:
    print(f"  - {name}")

print(f"\nМета-модель: {meta_learner.__class__.__name__}")


# 6. МОДЕЛЬ 2: МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLP)

print("\n" + "="*80)
print("5. МОДЕЛЬ 2: МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLPClassifier)")
print("="*80)

# Создание MLP с двумя скрытыми слоями
mlp_model = MLPClassifier(
    hidden_layer_sizes=(10, 5),  # 2 скрытых слоя: 10 и 5 нейронов
    activation='relu',            # функция активации
    solver='adam',                # оптимизатор
    alpha=0.0001,                 # коэффициент регуляризации
    max_iter=1000,                # максимальное количество итераций
    random_state=42,
    verbose=False
)

# Обучение
mlp_model.fit(X_train, y_train)

# Предсказание
y_pred_mlp = mlp_model.predict(X_test)

# Оценка качества
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp)

print(f"Результаты модели MLP:")
print(f"  Accuracy = {acc_mlp:.4f} ({acc_mlp*100:.2f}%)")
print(f"  F1-score = {f1_mlp:.4f}")

print(f"\nАрхитектура MLP:")
print(f"  Входной слой: {X_train.shape[1]} нейронов")
print(f"  Скрытые слои: {mlp_model.hidden_layer_sizes}")
print(f"  Выходной слой: 1 нейрон (бинарная классификация)")
print(f"  Функция активации: {mlp_model.activation}")
print(f"  Оптимизатор: {mlp_model.solver}")
print(f"  Количество итераций при обучении: {mlp_model.n_iter_}")

# График функции потерь (если доступен)
if hasattr(mlp_model, 'loss_curve_') and len(mlp_model.loss_curve_) > 0:
    plt.figure(figsize=(8, 5))
    plt.plot(mlp_model.loss_curve_, 'b-', linewidth=2)
    plt.xlabel('Итерация', fontsize=12)
    plt.ylabel('Значение функции потерь', fontsize=12)
    plt.title('Кривая обучения MLP', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 7. МОДЕЛЬ 3: МГУА (GMDH) - если доступно

print("\n" + "="*80)
print("6. МОДЕЛЬ 3: МГУА (Метод группового учета аргументов)")
print("="*80)

gmdh_results = {}

if GMDH_AVAILABLE:
    try:
        # Так как библиотека gmdh предназначена для регрессии,
        # а у нас задача классификации, преобразуем задачу
        # Также библиотека требует данные в формате list
        
        # Преобразуем данные в список списков
        X_train_list = X_train.tolist()
        X_test_list = X_test.tolist()
        y_train_list = y_train.tolist()
        
        # 7.1 Линейный метод: COMBI (комбинаторный алгоритм)
        print("\n7.1 МГУА: COMBI (комбинаторный алгоритм)")
        print("-"*40)
        
        try:
            from gmdh import Combi, split_data
            combi_model = Combi()
            combi_model.fit(X_train_list, y_train_list)
            y_pred_combi = combi_model.predict(X_test_list)
            
            # Для классификации округляем предсказания
            y_pred_combi_class = [round(abs(x)) for x in y_pred_combi]
            y_pred_combi_class = [1 if x >= 1 else 0 for x in y_pred_combi_class]
            
            acc_combi = accuracy_score(y_test, y_pred_combi_class)
            f1_combi = f1_score(y_test, y_pred_combi_class)
            
            gmdh_results['COMBI'] = {'Accuracy': acc_combi, 'F1-score': f1_combi}
            
            print(f"  Результаты COMBI:")
            print(f"    Accuracy = {acc_combi:.4f} ({acc_combi*100:.2f}%)")
            print(f"    F1-score = {f1_combi:.4f}")
            
            # Вывод полинома
            try:
                best_poly = combi_model.get_best_polynomial()
                print(f"    Найденная модель: {best_poly}")
            except:
                pass
        except Exception as e:
            print(f"  Ошибка при обучении COMBI: {e}")
        
        # 7.2 Нелинейный метод: MIA (итерационный многослойный алгоритм)
        print("\n7.2 МГУА: MIA (итерационный многослойный алгоритм)")
        print("-"*40)
        
        try:
            from gmdh import MIA
            mia_model = MIA()
            mia_model.fit(X_train_list, y_train_list)
            y_pred_mia = mia_model.predict(X_test_list)
            
            y_pred_mia_class = [round(abs(x)) for x in y_pred_mia]
            y_pred_mia_class = [1 if x >= 1 else 0 for x in y_pred_mia_class]
            
            acc_mia = accuracy_score(y_test, y_pred_mia_class)
            f1_mia = f1_score(y_test, y_pred_mia_class)
            
            gmdh_results['MIA'] = {'Accuracy': acc_mia, 'F1-score': f1_mia}
            
            print(f"  Результаты MIA:")
            print(f"    Accuracy = {acc_mia:.4f} ({acc_mia*100:.2f}%)")
            print(f"    F1-score = {f1_mia:.4f}")
        except Exception as e:
            print(f"  Ошибка при обучении MIA: {e}")
            
    except Exception as e:
        print(f"\nОшибка при работе с библиотекой gmdh: {e}")
        print("Библиотека gmdh может не поддерживать задачу классификации.")
        print("Согласно документации, библиотека предназначена для решения задач регрессии.")
else:
    print("Библиотека gmdh не установлена.")
    print("МГУА модели не были обучены.")
    print("\nДля установки библиотеки выполните команду:")
    print("  pip install gmdh")
    print("\nПримечание: Библиотека gmdh предназначена для решения задач РЕГРЕССИИ.")
    print("Для бинарной классификации требуется дополнительная обработка результатов.")
    print("Также существуют аналогичные библиотеки для классификации (например, GMDH2 для R).")

# ============================================================
# 8. СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ
# ============================================================

print("\n" + "="*80)
print("7. СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ")
print("="*80)

# Сбор результатов всех моделей
all_results = {
    'Стекинг': {'Accuracy': acc_stacking, 'F1-score': f1_stacking},
    'MLP': {'Accuracy': acc_mlp, 'F1-score': f1_mlp}
}

# Добавляем результаты МГУА, если они есть
all_results.update(gmdh_results)

# Вывод сводной таблицы
print("\nСводная таблица результатов:")
print(f"\n{'Модель':<20} {'Accuracy':<12} {'F1-score':<12}")
print("-" * 50)

for name, scores in all_results.items():
    print(f"{name:<20} {scores['Accuracy']:<12.4f} {scores['F1-score']:<12.4f}")

# Определение лучшей модели по F1-score
best_model = max(all_results, key=lambda x: all_results[x]['F1-score'])
best_score = all_results[best_model]['F1-score']

print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model} (F1-score = {best_score:.4f})")

# ============================================================
# 9. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ МОДЕЛЕЙ
# ============================================================

print("\n8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-"*50)

# График сравнения моделей
plt.figure(figsize=(12, 6))

models_names = list(all_results.keys())
accuracy_scores = [all_results[m]['Accuracy'] for m in models_names]
f1_scores = [all_results[m]['F1-score'] for m in models_names]

x = np.arange(len(models_names))
width = 0.35

bars1 = plt.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='steelblue', edgecolor='black')
bars2 = plt.bar(x + width/2, f1_scores, width, label='F1-score', color='coral', edgecolor='black')

plt.xlabel('Модели', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.title('Сравнение качества моделей', fontsize=14, fontweight='bold')
plt.xticks(x, models_names, rotation=15, ha='right')
plt.legend()
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3, axis='y')

# Добавление значений на столбцы
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================
# 10. МАТРИЦЫ ОШИБОК ДЛЯ ВСЕХ МОДЕЛЕЙ
# ============================================================

print("\n9. МАТРИЦЫ ОШИБОК")
print("-"*50)

# Создаем матрицы ошибок для каждой модели
models_for_cm = [
    (stacking_model, 'Стекинг', y_pred_stacking),
    (mlp_model, 'MLP', y_pred_mlp)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cmaps = ['Blues', 'Greens']

for idx, (model, name, y_pred) in enumerate(models_for_cm):
    cm = confusion_matrix(y_test, y_pred)
    im = axes[idx].imshow(cm, interpolation='nearest', cmap=cmaps[idx])
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}', fontsize=12)
    axes[idx].set_xlabel('Предсказано')
    axes[idx].set_ylabel('Фактически')
    
    # Добавляем значения в ячейки
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    
    axes[idx].set_xticks([0, 1])
    axes[idx].set_yticks([0, 1])
    axes[idx].set_xticklabels(['Не сдал', 'Сдал'])
    axes[idx].set_yticklabels(['Не сдал', 'Сдал'])

plt.colorbar(im, ax=axes[1])
plt.suptitle('Матрицы ошибок моделей', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 11. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: СРАВНЕНИЕ С БЭГГИНГОМ И БУСТИНГОМ
# ============================================================

print("\n10. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: СРАВНЕНИЕ С ДРУГИМИ АНСАМБЛЯМИ")
print("-"*50)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Бэггинг: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Бустинг: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

# Бустинг: AdaBoost
ab = AdaBoostClassifier(n_estimators=100, random_state=42)
ab.fit(X_train, y_train)
y_pred_ab = ab.predict(X_test)
acc_ab = accuracy_score(y_test, y_pred_ab)
f1_ab = f1_score(y_test, y_pred_ab)

print("\nСравнение с другими ансамблевыми методами:")
print(f"\n{'Метод':<20} {'Accuracy':<12} {'F1-score':<12}")
print("-"*50)
print(f"{'Бэггинг (RF)':<20} {acc_rf:<12.4f} {f1_rf:<12.4f}")
print(f"{'Бустинг (GB)':<20} {acc_gb:<12.4f} {f1_gb:<12.4f}")
print(f"{'Бустинг (AdaBoost)':<20} {acc_ab:<12.4f} {f1_ab:<12.4f}")

# ============================================================
# 12. ИТОГОВЫЕ ВЫВОДЫ
# ============================================================

print("\n" + "="*80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("="*80)

print(f"""
1. ЗАДАЧА: Бинарная классификация (предсказание сдачи экзамена)

2. РАЗМЕР ВЫБОРКИ:
   • Всего студентов: {len(df)}
   • Обучающая выборка: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.0f}%)
   • Тестовая выборка: {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.0f}%)

3. РЕЗУЛЬТАТЫ ПО МЕТРИКЕ ACCURACY:
""")
for name, scores in all_results.items():
    print(f"   • {name}: {scores['Accuracy']:.4f}")

print(f"""
4. РЕЗУЛЬТАТЫ ПО МЕТРИКЕ F1-SCORE:
""")
for name, scores in all_results.items():
    print(f"   • {name}: {scores['F1-score']:.4f}")

print(f"""
5. ЛУЧШАЯ МОДЕЛЬ: {best_model} (F1-score = {best_score:.4f})

6. ВЫВОДЫ ПО МОДЕЛЯМ:

   • СТЕКИНГ:
     - Комбинирует несколько базовых моделей для улучшения качества
     - Использует мета-модель для обучения на предсказаниях базовых моделей
     - Эффективен при разнообразии базовых алгоритмов

   • MLP (МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН):
     - Простая нейронная сеть прямого распространения
     - Способен выявлять нелинейные зависимости
     - Чувствителен к масштабированию данных

   • МГУА:
     - Метод группового учета аргументов для самоорганизации моделей
     - Основное применение - решение задач РЕГРЕССИИ
     - Для классификации требуется дополнительная обработка результатов

7. СРАВНЕНИЕ С ДРУГИМИ АНСАМБЛЯМИ:
   • Бэггинг (Random Forest) показывает устойчивые результаты
   • Бустинг (Gradient Boosting) эффективен для сложных зависимостей
   • Стекинг может превзойти отдельные модели за счет комбинации

8. РЕКОМЕНДАЦИИ:
   • Для данного набора данных рекомендуется использовать {best_model}
   • Для улучшения качества можно провести настройку гиперпараметров
   • Для задач классификации МГУА требует адаптации или использования специализированных версий
""")

print("\n" + "="*80)
print("ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА")
print("="*80)