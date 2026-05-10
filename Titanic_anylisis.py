import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv('Titanic-Dataset.csv')

print("ПЕРВЫЕ 5 СТРОК ДАТАСЕТА")
print(df.head())
print("\n")

# общая информация о наборе данных
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")
print("\n")
print("Названия столбцов:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
print("\n")
print("Типы данных столбцов:")
print(df.dtypes)
print("\n")

print("статическое описание числовых столбцов:")
print(df.describe())
print("\n")

# Анализ пропущенных значений

# Количество пропущенных значений
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Кол-во пропусков': missing_values,
    'Процент от общего': missing_percentage.round(2)
})
missing_df = missing_df[missing_df['Кол-во пропусков'] > 0].sort_values('Кол-во пропусков', ascending=False)

print("Пропущенные значения в столбцах:")
print(missing_df)
print("\n")

# Визуализация пропущенных значений
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Карта пропущенных значений', fontsize=16)
plt.xlabel('Столбцы')
plt.ylabel('Строки')
plt.tight_layout()
plt.show()

# Схема заполнения пропущенных значений

# Создаем копию для обработки
df_clean = df.copy()
print("Порт посадки:")
print("Метод: Модальное заполнение (самое частое значение)")
most_common_port = df_clean['Embarked'].mode()[0]
print(f"Самый частый порт посадки: {most_common_port}")
print("Обоснование: Пропусков всего 2, мода не исказит статистику")

df_clean['Embarked'].fillna(most_common_port, inplace=True)
print(f"Пропусков после заполнения: {df_clean['Embarked'].isnull().sum()}")
print("\n")

print("Заполнение возраста:")
print("Метод: Заполнение медианой по группам (Pclass и Sex)")
print("Обоснование: Возраст коррелирует с классом и полом, медиана устойчива к выбросам")

# Вычисляем медиану возраста для каждой группы
age_medians = df_clean.groupby(['Pclass', 'Sex'])['Age'].median()
print("Медианы возраста по группам (Pclass, Sex):")
print(age_medians)
print("\n")

def fill_age(row):
    if pd.isnull(row['Age']):
        return age_medians[row['Pclass'], row['Sex']]
    else:
        return row['Age']

df_clean['Age'] = df_clean.apply(fill_age, axis=1)
print(f"Пропусков Age после заполнения: {df_clean['Age'].isnull().sum()}")
print("\n")

# Обработка Cabin (каюта)
print("Обработка кают:")
print("Метод: Создание бинарного признака Cabin_known")
print("Обоснование: Пропусков >75%, заполнять бессмысленно. Отсутствие каюты может быть признаком")

# Создаем новый признак
df_clean['Cabin_known'] = df_clean['Cabin'].notnull().astype(int)

print(f"Распределение Cabin_known:")
print(f"0 (каюта неизвестна): {(df_clean['Cabin_known'] == 0).sum()} пассажиров")
print(f"1 (каюта известна): {(df_clean['Cabin_known'] == 1).sum()} пассажиров")
print("\n")

print("анализ выживаемости")

# Общее количество выживших
total_survived = df_clean['Survived'].sum()
total_passengers = len(df_clean)
survival_rate = (total_survived / total_passengers) * 100

print(f"Всего пассажиров: {total_passengers}")
print(f"Всего выживших: {total_survived}")
print(f"Всего погибших: {total_passengers - total_survived}")
print(f"Общий процент выживаемости: {survival_rate:.2f}%")
print("\n")

# По полу
print("Выживаемость по полу:")
survival_by_sex = df_clean.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_sex['percent'] = survival_by_sex['mean'] * 100
print(survival_by_sex)
print("\n")

# По классу обслуживания
print("Выживаемость по классу:")
survival_by_class = df_clean.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_class['percent'] = survival_by_class['mean'] * 100
print(survival_by_class)
print("\n")

# По возрастным категориям
print("Выживаемость по возрасту:")

# Создаем возрастные группы
bins = [0, 7, 12, 18, 25, 35, 45, 55, 65, 120]
labels = ['0-7 лет', '7-12 лет', '12-18 лет', '18-25 лет', 
          '25-35 лет', '35-45 лет', '45-55 лет', '55-65 лет', '65+ лет']

df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=bins, labels=labels, right=False)

# Статистика по группам
age_group_stats = df_clean.groupby('AgeGroup', observed=True)['Survived'].agg(['count', 'sum', 'mean'])
age_group_stats['percent'] = age_group_stats['mean'] * 100
print(age_group_stats)
print("\n")

# Средний возраст в каждой группе
print("Средний возраст в каждой группе:")
avg_age_by_group = df_clean.groupby('AgeGroup', observed=True)['Age'].mean().round(1)
print(avg_age_by_group)
print("\n")

# пол + класс
print("Анализ пола + класса:")
survival_combined = df_clean.groupby(['Sex', 'Pclass'])['Survived'].agg(['count', 'sum', 'mean'])
survival_combined['percent'] = survival_combined['mean'] * 100
print(survival_combined)
print("\n")

# Визуализация

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Анализ выживаемости на Титанике', fontsize=16)

# Общая выживаемость
axes[0, 0].pie([total_survived, total_passengers - total_survived], 
               labels=['Выжили', 'Погибли'], 
               autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'],
               startangle=90)
axes[0, 0].set_title('Общая выживаемость', fontsize=14)

# Выживаемость по полу
sns.barplot(x='Sex', y='Survived', data=df_clean, ax=axes[0, 1])
axes[0, 1].set_title('Выживаемость по полу', fontsize=14)
axes[0, 1].set_ylabel('Доля выживших')
axes[0, 1].set_xlabel('Пол')

# Выживаемость по классу
sns.barplot(x='Pclass', y='Survived', data=df_clean, ax=axes[0, 2])
axes[0, 2].set_title('Выживаемость по классу', fontsize=14)
axes[0, 2].set_ylabel('Доля выживших')
axes[0, 2].set_xlabel('Класс')

# Выживаемость по возрастным группам
sns.barplot(x='AgeGroup', y='Survived', data=df_clean, ax=axes[1, 0], 
            order=labels)
axes[1, 0].set_title('Выживаемость по возрастным группам', fontsize=14)
axes[1, 0].set_ylabel('Доля выживших')
axes[1, 0].set_xlabel('Возрастная группа')
axes[1, 0].tick_params(axis='x', rotation=45)

# Распределение возрастов
axes[1, 1].hist([df_clean[df_clean['Survived']==1]['Age'], 
                 df_clean[df_clean['Survived']==0]['Age']], 
                bins=20, label=['Выжили', 'Погибли'], 
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Распределение возрастов', fontsize=14)
axes[1, 1].set_xlabel('Возраст')
axes[1, 1].set_ylabel('Количество')
axes[1, 1].legend()

# Тепловая карта: пол + класс
pivot_table = df_clean.pivot_table('Survived', index='Sex', columns='Pclass', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 2], 
            cbar_kws={'label': 'Доля выживших'})
axes[1, 2].set_title('Выживаемость: Пол × Класс', fontsize=14)

plt.tight_layout()
plt.show()

# Ищем людей без имени

# Проверяем на пустые строки
df_clean['Name_cleaned'] = df_clean['Name'].astype(str).str.strip()
empty_names = df_clean[df_clean['Name_cleaned'] == '']
print(f"Пустых строк в имени: {len(empty_names)}")

# Удаляем людей без имени (если есть)
original_len = len(df_clean)
df_clean = df_clean.dropna(subset=['Name'])
df_clean = df_clean[df_clean['Name_cleaned'] != '']