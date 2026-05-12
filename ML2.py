import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("ЛАБОРАТОРНАЯ РАБОТА №1: Исследование и визуализация данных")
print("="*80)

# ============================================================
# РАЗДЕЛ 1: ЗАГРУЗКА И ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ
# ============================================================

print("\n" + "="*80)
print("РАЗДЕЛ 1: ЗАГРУЗКА И ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ")
print("="*80)

# Загрузка данных
df = pd.read_csv('успеваемость.csv')
print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

print("\nПервые 5 строк:")
print(df.head())

print("\nПоследние 5 строк:")
print(df.tail())

print("\nТипы данных:")
print(df.dtypes)

print("\nОбщая информация:")
print(df.info())

print("\nСтатистическое описание числовых признаков:")
print(df.describe())

# ============================================================
# РАЗДЕЛ 2: ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАТАСЕТА
# ============================================================

print("\n" + "="*80)
print("РАЗДЕЛ 2: ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАТАСЕТА")
print("="*80)

print(f"""
1. ОБЩАЯ ИНФОРМАЦИЯ:
   • Количество наблюдений: {len(df)} студентов
   • Количество признаков: {len(df.columns)} (включая целевой)
   • Количество числовых признаков: {len(df.select_dtypes(include=['int64', 'float64']).columns)}
   • Количество категориальных признаков: {len(df.select_dtypes(include=['object']).columns)}

2. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ('сдал_экзамен'):
   • Уникальные значения: {df['сдал_экзамен'].unique()}
   • Распределение: Сдало - {df['сдал_экзамен'].sum()} студентов ({(df['сдал_экзамен'].sum()/len(df)*100):.1f}%)
                   Не сдало - {(len(df)-df['сдал_экзамен'].sum())} студентов ({(len(df)-df['сдал_экзамен'].sum())/len(df)*100:.1f}%)

3. ПРИЗНАКИ:
""")

for col in df.columns:
    if col != 'сдал_экзамен' and col != 'студент_id':
        print(f"   • {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, std={df[col].std():.2f}")

# ============================================================
# РАЗДЕЛ 3: ВИЗУАЛЬНОЕ ИССЛЕДОВАНИЕ ДАТАСЕТА
# ============================================================

print("\n" + "="*80)
print("РАЗДЕЛ 3: ВИЗУАЛЬНОЕ ИССЛЕДОВАНИЕ ДАТАСЕТА")
print("="*80)

# 3.1 Гистограммы распределения признаков
print("\n3.1 ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['часы_занятий', 'процент_прогулов', 'средний_балл', 'выполнено_дз']
colors = ['steelblue', 'coral', 'seagreen', 'purple']

for ax, feature, color in zip(axes.flatten(), features, colors):
    ax.hist(df[feature], bins=8, edgecolor='black', alpha=0.7, color=color)
    ax.set_title(f'Распределение признака: {feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('Частота', fontsize=10)
    ax.axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {df[feature].mean():.1f}')
    ax.axvline(df[feature].median(), color='green', linestyle='--', linewidth=2, label=f'Медиана: {df[feature].median():.1f}')
    ax.legend()

plt.suptitle('Гистограммы распределения признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('гистограммы_признаков.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.2 Распределение целевой переменной
print("\n3.2 РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("-"*50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Круговая диаграмма
labels = ['Сдал экзамен', 'Не сдал экзамен']
sizes = [df['сдал_экзамен'].sum(), len(df) - df['сдал_экзамен'].sum()]
colors = ['#2E8B57', '#CD5C5C']
explode = (0.05, 0)

axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
axes[0].set_title('Распределение студентов по сдаче экзамена', fontsize=12, fontweight='bold')

# Столбчатая диаграмма
sns.countplot(data=df, x='сдал_экзамен', palette=['#CD5C5C', '#2E8B57'], ax=axes[1])
axes[1].set_title('Количество студентов, сдавших и не сдавших экзамен', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Результат экзамена (0 - не сдал, 1 - сдал)', fontsize=10)
axes[1].set_ylabel('Количество студентов')
for i, v in enumerate([len(df)-df['сдал_экзамен'].sum(), df['сдал_экзамен'].sum()]):
    axes[1].text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('распределение_целевой.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.3 Ящики с усами (Boxplot) для выявления выбросов
print("\n3.3 ЯЩИКИ С УСАМИ (BOXPLOT) ДЛЯ ВЫЯВЛЕНИЯ ВЫБРОСОВ")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, feature, color in zip(axes.flatten(), features, colors):
    sns.boxplot(data=df, y=feature, color=color, ax=ax)
    ax.set_title(f'Boxplot признака: {feature}', fontsize=12, fontweight='bold')
    ax.set_ylabel(feature, fontsize=10)

    # Подсчет выбросов
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        ax.text(0.02, 0.98, f'Выбросов: {len(outliers)}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, color='red', fontweight='bold')

plt.suptitle('Ящики с усами для выявления выбросов', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplot_выбросы.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.4 Диаграмма рассеяния (Scatter plot)
print("\n3.4 ДИАГРАММА РАССЕЯНИЯ (Scatter plot)")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Разделяем на сдавших и не сдавших
df_passed = df[df['сдал_экзамен'] == 1]
df_failed = df[df['сдал_экзамен'] == 0]

# График 1: часы_занятий vs средний_балл
axes[0,0].scatter(df_passed['часы_занятий'], df_passed['средний_балл'], color='green', s=80, alpha=0.7, label='Сдал')
axes[0,0].scatter(df_failed['часы_занятий'], df_failed['средний_балл'], color='red', s=80, alpha=0.7, label='Не сдал')
axes[0,0].set_xlabel('Часы занятий', fontsize=10)
axes[0,0].set_ylabel('Средний балл', fontsize=10)
axes[0,0].set_title('Часы занятий vs Средний балл', fontsize=12, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# График 2: выполнено_дз vs средний_балл
axes[0,1].scatter(df_passed['выполнено_дз'], df_passed['средний_балл'], color='green', s=80, alpha=0.7, label='Сдал')
axes[0,1].scatter(df_failed['выполнено_дз'], df_failed['средний_балл'], color='red', s=80, alpha=0.7, label='Не сдал')
axes[0,1].set_xlabel('Выполнено ДЗ (%)', fontsize=10)
axes[0,1].set_ylabel('Средний балл', fontsize=10)
axes[0,1].set_title('Выполнено ДЗ vs Средний балл', fontsize=12, fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# График 3: часы_занятий vs выполнено_дз
axes[1,0].scatter(df_passed['часы_занятий'], df_passed['выполнено_дз'], color='green', s=80, alpha=0.7, label='Сдал')
axes[1,0].scatter(df_failed['часы_занятий'], df_failed['выполнено_дз'], color='red', s=80, alpha=0.7, label='Не сдал')
axes[1,0].set_xlabel('Часы занятий', fontsize=10)
axes[1,0].set_ylabel('Выполнено ДЗ (%)', fontsize=10)
axes[1,0].set_title('Часы занятий vs Выполнено ДЗ', fontsize=12, fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# График 4: процент_прогулов vs средний_балл
axes[1,1].scatter(df_passed['процент_прогулов'], df_passed['средний_балл'], color='green', s=80, alpha=0.7, label='Сдал')
axes[1,1].scatter(df_failed['процент_прогулов'], df_failed['средний_балл'], color='red', s=80, alpha=0.7, label='Не сдал')
axes[1,1].set_xlabel('Процент прогулов (%)', fontsize=10)
axes[1,1].set_ylabel('Средний балл', fontsize=10)
axes[1,1].set_title('Процент прогулов vs Средний балл', fontsize=12, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Диаграммы рассеяния признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('диаграммы_рассеяния.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# РАЗДЕЛ 4: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================================

print("\n" + "="*80)
print("РАЗДЕЛ 4: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("="*80)

# 4.1 Корреляционная матрица
print("\n4.1 КОРРЕЛЯЦИОННАЯ МАТРИЦА")
print("-"*50)

# Вычисляем корреляцию всех признаков (исключая студент_id)
corr_matrix = df.drop('студент_id', axis=1).corr()

print("\nКорреляция признаков с целевой переменной 'сдал_экзамен':")
print("-"*45)
corr_with_target = corr_matrix['сдал_экзамен'].sort_values(ascending=False)
for feature, corr in corr_with_target.items():
    if feature != 'сдал_экзамен':
        print(f"  {feature}: {corr:.3f}")

print("\nКорреляционная матрица (все признаки):")
print("-"*45)
print(corr_matrix.round(3))

# 4.2 Тепловая карта корреляции
print("\n4.2 ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИИ")
print("-"*50)

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхней половины
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8})
plt.title('Тепловая карта корреляции признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('тепловая_карта_корреляции.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.3 Анализ корреляции
print("\n4.3 АНАЛИЗ КОРРЕЛЯЦИИ (выводы)")
print("-"*50)

print("""
ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ КОРРЕЛЯЦИИ:
   |r| ≈ 0.9-1.0 → очень сильная корреляция
   |r| ≈ 0.7-0.9 → сильная корреляция
   |r| ≈ 0.5-0.7 → умеренная корреляция
   |r| ≈ 0.3-0.5 → слабая корреляция
   |r| ≈ 0.0-0.3 → очень слабая/отсутствует
   
   Положительная корреляция (близка к +1):
       При увеличении одного признака растет и другой
   
   Отрицательная корреляция (близка к -1):
       При увеличении одного признака другой уменьшается
""")

# ============================================================
# РАЗДЕЛ 5: ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ
# ============================================================

print("\n" + "="*80)
print("РАЗДЕЛ 5: ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ")
print("="*80)

# 5.1 Violin plot (скрипичный график) - показывает распределение и плотность
print("\n5.1 СКРИПИЧНЫЙ ГРАФИК (Violin plot)")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, feature, color in zip(axes.flatten(), features, colors):
    sns.violinplot(data=df, x='сдал_экзамен', y=feature, palette=['#CD5C5C', '#2E8B57'], ax=ax)
    ax.set_title(f'Распределение {feature} по группам студентов', fontsize=11, fontweight='bold')
    ax.set_xlabel('Результат экзамена (0 - не сдал, 1 - сдал)', fontsize=9)
    ax.set_ylabel(feature, fontsize=9)

plt.suptitle('Скрипичные графики распределения признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('violin_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# 5.2 Pairplot (матрица диаграмм рассеяния) - для всех пар признаков
print("\n5.2 МАТРИЦА ДИАГРАММ РАССЕЯНИЯ (Pairplot)")
print("-"*50)

# Если данных много, можно отобразить основные признаки
sns.pairplot(df, hue='сдал_экзамен', vars=['часы_занятий', 'процент_прогулов', 'средний_балл', 'выполнено_дз'],
             palette={0: '#CD5C5C', 1: '#2E8B57'}, diag_kind='hist')
plt.suptitle('Матрица диаграмм рассеяния всех признаков', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# 5.3 Распределение сдавших и не сдавших по каждому признаку
print("\n5.3 СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ПО ГРУППАМ")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, feature, color in zip(axes.flatten(), features, colors):
    sns.histplot(data=df, x=feature, hue='сдал_экзамен', bins=8, alpha=0.6, palette={0: 'red', 1: 'green'}, ax=ax)
    ax.set_title(f'Распределение {feature} для сдавших и не сдавших', fontsize=11, fontweight='bold')
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('Частота', fontsize=10)

plt.suptitle('Сравнение распределений признаков по группам', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('сравнение_распределений.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# РАЗДЕЛ 6: ИТОГОВЫЕ ВЫВОДЫ
# ============================================================

print("\n" + "="*80)
print("РАЗДЕЛ 6: ИТОГОВЫЕ ВЫВОДЫ ПО РЕЗУЛЬТАТАМ АНАЛИЗА")
print("="*80)

print("""
1. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:
""")
for feature, corr in corr_with_target.items():
    if feature != 'сдал_экзамен':
        if abs(corr) >= 0.7:
            strength = "СИЛЬНАЯ"
        elif abs(corr) >= 0.5:
            strength = "УМЕРЕННАЯ"
        elif abs(corr) >= 0.3:
            strength = "СЛАБАЯ"
        else:
            strength = "ОЧЕНЬ СЛАБАЯ"
        sign = "положительная" if corr > 0 else "отрицательная"
        print(f"   • {feature}: {strength} {sign} корреляция (коэф. = {corr:.3f})")

print("""
2. НАИБОЛЕЕ ВАЖНЫЕ ПРИЗНАКИ ДЛЯ ПРЕДСКАЗАНИЯ:
   - Наибольшее влияние на сдачу экзамена оказывает средний балл
   - Часы занятий и процент выполненных ДЗ также важны
   - Процент прогулов имеет обратную зависимость

3. ОСОБЕННОСТИ РАСПРЕДЕЛЕНИЯ:
   - Распределение студентов по успеваемости близко к нормальному
   - Выбросов в данных не обнаружено (или их очень мало)
   - Классы сбалансированы (50%/50%)

4. ПРАКТИЧЕСКИЕ ВЫВОДЫ:
   - Для повышения успеваемости нужно работать над средним баллом
   - Увеличение часов занятий и выполнения ДЗ положительно влияет на результат
   - Модели машинного обучения на этих данных могут показать хорошее качество
""")

print("\n" + "="*80)
print("ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА")
print("Сохраненные файлы:")
print("  • гистограммы_признаков.png")
print("  • распределение_целевой.png")
print("  • boxplot_выбросы.png")
print("  • диаграммы_рассеяния.png")
print("  • тепловая_карта_корреляции.png")
print("  • violin_plot.png")
print("  • pairplot.png")
print("  • сравнение_распределений.png")
print("="*80)