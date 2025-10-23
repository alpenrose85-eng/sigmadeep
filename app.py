import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import io

# Данные из таблицы ГОСТ 5639-82
grain_data = {
    'grain_size': [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'grain_area': [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.00781, 0.00390, 
                   0.00195, 0.00098, 0.00049, 0.000244, 0.000122, 0.000061, 0.000030, 
                   0.000015, 0.000008],
    'grain_diameter': [0.875, 0.650, 0.444, 0.313, 0.222, 0.157, 0.111, 0.0783, 0.0553,
                       0.0391, 0.0267, 0.0196, 0.0138, 0.0099, 0.0069, 0.0049, 0.0032, 0.0027]
}

grain_df = pd.DataFrame(grain_data)

def enhanced_growth_model(t, k, n, grain_area, alpha=0.5, d0=0):
    """
    Улучшенная модель роста с учетом площади зерна
    d = d0 + k * (1 + alpha/grain_area) * t^n
    где alpha - коэффициент влияния границ зерен
    """
    boundary_effect = 1 + alpha / grain_area
    return d0 + k * boundary_effect * (t ** n)

def boundary_density_model(t, k, n, grain_area, beta=0.1, d0=0):
    """
    Альтернативная модель: влияние через плотность границ
    d = d0 + k * (1 + beta * (1/grain_area)) * t^n
    """
    boundary_density_effect = 1 + beta * (1 / grain_area)
    return d0 + k * boundary_density_effect * (t ** n)

# Основная программа Streamlit
st.title("Улучшенная модель роста σ-фазы с учетом размера зерна")

# Показываем таблицу ГОСТ
with st.expander("Данные ГОСТ 5639-82 о размерах зерен"):
    st.dataframe(grain_df)
    st.markdown("""
    **Ключевая идея:** Меньшая площадь зерна → больше границ зерен → больше мест зарождения σ-фазы → ускоренный рост
    """)

# Загрузка данных
st.subheader("Загрузка экспериментальных данных")

# Создаем шаблон Excel файла для загрузки
def create_template():
    template_data = {
        'G': [7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3],
        'T': [600, 600, 600, 600, 650, 650, 650, 650, 700, 700, 700, 700],
        't': [2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000],
        'd': [2.1, 3.0, 3.6, 4.1, 3.5, 4.8, 5.8, 6.5, 5.2, 7.1, 8.5, 9.6]
    }
    df_template = pd.DataFrame(template_data)
    return df_template

# Кнопка для скачивания шаблона
template_df = create_template()

# Конвертируем DataFrame в Excel файл в памяти
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    template_df.to_excel(writer, sheet_name='Шаблон_данных', index=False)
    # Добавляем лист с описанием
    description_df = pd.DataFrame({
        'Колонка': ['G', 'T', 't', 'd'],
        'Описание': [
            'Номер зерна по ГОСТ 5639-82',
            'Температура в °C', 
            'Время выдержки в часах',
            'Эквивалентный диаметр σ-фазы в мкм'
        ],
        'Пример': ['7, 5, 3', '600, 650, 700', '2000, 4000, 6000, 8000', '2.1, 3.0, 4.1']
    })
    description_df.to_excel(writer, sheet_name='Описание_колонок', index=False)

excel_buffer.seek(0)

st.download_button(
    label="📥 Скачать шаблон Excel файла",
    data=excel_buffer,
    file_name="шаблон_данных_сигма_фаза.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Скачайте шаблон для заполнения вашими экспериментальными данными"
)

# Загрузка файла
uploaded_file = st.file_uploader(
    "Загрузите файл с экспериментальными данными", 
    type=['csv', 'xlsx', 'xls'],
    help="Поддерживаемые форматы: CSV, Excel (.xlsx, .xls)"
)

df = None

if uploaded_file is not None:
    try:
        # Определяем тип файла и загружаем соответствующим образом
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.success("CSV файл успешно загружен!")
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Для Excel файлов показываем доступные листы
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                st.success(f"Excel файл загружен с листа: {sheet_names[0]}")
            else:
                selected_sheet = st.selectbox(
                    "Выберите лист с данными:",
                    options=sheet_names
                )
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                st.success(f"Данные загружены с листа: {selected_sheet}")
        
        # Проверяем необходимые колонки
        required_columns = ['G', 'T', 't', 'd']
        if all(col in df.columns for col in required_columns):
            st.success("Все необходимые колонки присутствуют!")
            
            # Показываем предпросмотр данных
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head())
            
            # Статистика по данным
            st.subheader("Статистика данных")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Количество записей", len(df))
            with col2:
                st.metric("Уникальные номера зерен", df['G'].nunique())
            with col3:
                st.metric("Температуры исследования", f"{df['T'].min()} - {df['T'].max()}°C")
            with col4:
                st.metric("Время выдержки", f"{df['t'].min()} - {df['t'].max()} ч")
            
            # Сохраняем данные в session_state
            st.session_state['experimental_data'] = df
            
        else:
            missing_columns = [col for col in required_columns if col not in df.columns]
            st.error(f"Отсутствуют необходимые колонки: {missing_columns}")
            st.info("Пожалуйста, используйте шаблон для правильного формата данных")
            
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        st.info("Убедитесь, что файл не поврежден и имеет правильный формат")

# Если данные загружены, продолжаем анализ
if 'experimental_data' in st.session_state:
    df = st.session_state['experimental_data']
    
    # Объединяем с данными о размере зерна
    df_enriched = df.merge(grain_df, left_on='G', right_on='grain_size', how='left')
    
    # Проверяем, есть ли соответствие номеров зерен
    unmatched_grains = df[~df['G'].isin(grain_df['grain_size'])]['G'].unique()
    if len(unmatched_grains) > 0:
        st.warning(f"Следующие номера зерен не найдены в базе ГОСТ: {list(unmatched_grains)}")
    
    # Анализ влияния размера зерна
    st.subheader("Анализ влияния размера зерна на рост σ-фазы")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Зависимость диаметра от времени для разных размеров зерна
    for grain_size in df_enriched['G'].unique():
        subset = df_enriched[df_enriched['G'] == grain_size]
        if not subset.empty and not pd.isna(subset['grain_area'].iloc[0]):
            grain_area = subset['grain_area'].iloc[0]
            label = f'Зерно {grain_size} (S={grain_area:.4f} мм²)'
            
            ax1.scatter(subset['t'], subset['d'], label=label, alpha=0.7)
            
            # Линия тренда
            if len(subset) > 1:
                z = np.polyfit(subset['t'], subset['d'], 1)
                p = np.poly1d(z)
                ax1.plot(subset['t'], p(subset['t']), linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Время (часы)')
    ax1.set_ylabel('Диаметр σ-фазы (мкм)')
    ax1.set_title('Влияние размера зерна на кинетику роста')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Зависимость скорости роста от площади зерна
    growth_rates = []
    grain_areas = []
    grain_sizes = []
    
    for grain_size in df_enriched['G'].unique():
        subset = df_enriched[df_enriched['G'] == grain_size]
        if len(subset) > 1 and not pd.isna(subset['grain_area'].iloc[0]):
            # Оценка скорости роста (производная)
            time_sorted = np.sort(subset['t'].unique())
            if len(time_sorted) >= 2:
                diameters = [subset[subset['t'] == t]['d'].mean() for t in time_sorted]
                growth_rate = (diameters[-1] - diameters[0]) / (time_sorted[-1] - time_sorted[0])
                growth_rates.append(growth_rate)
                grain_areas.append(subset['grain_area'].iloc[0])
                grain_sizes.append(grain_size)
    
    if growth_rates:
        ax2.scatter(grain_areas, growth_rates, s=80, alpha=0.7)
        
        # Добавляем подписи точек
        for i, (area, rate, size) in enumerate(zip(grain_areas, growth_rates, grain_sizes)):
            ax2.annotate(f'G{size}', (area, rate), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Площадь зерна (мм²)')
        ax2.set_ylabel('Скорость роста (мкм/час)')
        ax2.set_title('Зависимость скорости роста от площади зерна')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Линия тренда
        if len(growth_rates) > 1:
            z = np.polyfit(np.log(grain_areas), growth_rates, 1)
            x_trend = np.logspace(np.log10(min(grain_areas)), np.log10(max(grain_areas)), 100)
            y_trend = z[0] * np.log(x_trend) + z[1]
            ax2.plot(x_trend, y_trend, 'r--', alpha=0.7, label='Тренд')
            ax2.legend()
    
    st.pyplot(fig)
    
    # Кнопка для выгрузки обогащенных данных
    st.subheader("Выгрузка результатов")
    
    # Создаем обогащенный DataFrame для выгрузки
    output_df = df_enriched.copy()
    
    # Конвертируем в Excel для выгрузки
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        output_df.to_excel(writer, sheet_name='Обогащенные_данные', index=False)
        
        # Добавляем лист с параметрами модели, если они есть
        if 'enhanced_params' in st.session_state:
            params_df = pd.DataFrame([st.session_state['enhanced_params']])
            params_df.to_excel(writer, sheet_name='Параметры_модели', index=False)
    
    output_buffer.seek(0)
    
    st.download_button(
        label="📊 Выгрузить обогащенные данные в Excel",
        data=output_buffer,
        file_name="результаты_анализа_сигма_фаза.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Скачайте полные результаты анализа с параметрами модели"
    )

# Информация о поддерживаемых форматах
with st.expander("ℹ️ Информация о форматах файлов"):
    st.markdown("""
    **Поддерживаемые форматы:**
    
    - **CSV**: Текстовый файл с разделителями-запятыми
    - **Excel**: Файлы .xlsx, .xls (Microsoft Excel)
    
    **Требуемые колонки:**
    
    | Колонка | Описание | Пример |
    |---------|----------|---------|
    | G | Номер зерна по ГОСТ 5639-82 | 7, 5, 3 |
    | T | Температура в °C | 600, 650, 700 |
    | t | Время выдержки в часах | 2000, 4000, 8000 |
    | d | Диаметр σ-фазы в мкм | 2.1, 3.0, 4.1 |
    
    **Рекомендации:**
    - Используйте шаблон для гарантии правильного формата
    - Сохраняйте названия колонок как в шаблоне
    - Для Excel файлов данные должны быть на первом листе или укажите нужный лист
    """)
