import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
import io
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Универсальная газовая постоянная
R = 8.314  # Дж/(моль·К)

st.title("🔬 Анализ кинетики роста σ-фазы для зерна №10")
st.markdown("""
**Фокус на одном номере зерна для отладки модели:**
- Степенной закон роста: $d^n - d_0^n = K \\cdot t$
- Температурная зависимость по Аррениусу: $K = K_0 \\cdot \\exp(-Q/RT)$
- Подбор показателя n в диапазоне 3.0-5.0
""")

# Загрузка данных
st.header("1. Загрузка данных для зерна №10")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                 value=0.0, min_value=0.0, step=0.1,
                                 help="Диаметр на минимальной наработке или близкий к 0")

target_grain = 10  # Фокусируемся на зерне №10

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if all(col in df.columns for col in ['G', 'T', 't', 'd']):
            # Фильтруем данные для зерна №10
            df_grain10 = df[df['G'] == target_grain].copy()
            
            if len(df_grain10) > 0:
                st.session_state['grain10_data'] = df_grain10
                st.success(f"✅ Данные для зерна №10 успешно загружены!")
                
                # Статистика по зерну №10
                st.subheader("📊 Статистика данных для зерна №10:")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    unique_temps = df_grain10['T'].unique()
                    st.metric("Температуры", f"{len(unique_temps)} уровней")
                    st.write(f"({', '.join(map(str, sorted(unique_temps)))}°C)")
                with col2:
                    time_range = f"{df_grain10['t'].min()} - {df_grain10['t'].max()}"
                    st.metric("Время выдержки", time_range + " ч")
                with col3:
                    diam_range = f"{df_grain10['d'].min():.1f} - {df_grain10['d'].max():.1f}"
                    st.metric("Диаметры", diam_range + " мкм")
                with col4:
                    st.metric("Всего точек", f"{len(df_grain10)}")
                
                # Детальная таблица по температурам
                st.write("**Распределение точек по температурам:**")
                temp_distribution = df_grain10.groupby('T').agg({
                    't': ['count', 'min', 'max'],
                    'd': ['min', 'max', 'mean']
                }).round(2)
                st.dataframe(temp_distribution)
                
                # Визуализация всех данных зерна №10
                st.subheader("📈 Визуализация данных зерна №10")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # График 1: Данные по температурам
                temperatures = sorted(df_grain10['T'].unique())
                colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
                
                for i, temp in enumerate(temperatures):
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    ax1.scatter(temp_data['t'], temp_data['d'], 
                               color=colors[i], label=f'{temp}°C', s=80, alpha=0.8)
                    
                    # Соединяем точки линиями для каждой температуры
                    sorted_data = temp_data.sort_values('t')
                    ax1.plot(sorted_data['t'], sorted_data['d'], 
                            color=colors[i], linestyle='--', alpha=0.5)
                
                ax1.set_xlabel('Время (часы)')
                ax1.set_ylabel('Диаметр σ-фазы (мкм)')
                ax1.set_title('Кинетика роста σ-фазы для зерна №10')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # График 2: Данные в логарифмических координатах
                for i, temp in enumerate(temperatures):
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    ax2.scatter(np.log(temp_data['t']), np.log(temp_data['d']), 
                               color=colors[i], label=f'{temp}°C', s=80, alpha=0.8)
                
                ax2.set_xlabel('ln(t)')
                ax2.set_ylabel('ln(d)')
                ax2.set_title('Данные в логарифмических координатах')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.dataframe(df_grain10)
                
            else:
                st.error(f"❌ В данных нет записей для зерна №10")
        else:
            missing = [col for col in ['G', 'T', 't', 'd'] if col not in df.columns]
            st.error(f"❌ Отсутствуют колонки: {missing}")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

# Основной расчет для зерна №10
if 'grain10_data' in st.session_state:
    df_grain10 = st.session_state['grain10_data']
    df_grain10['T_K'] = df_grain10['T'] + 273.15  # Температура в Кельвинах
    
    # Подбор показателя степени n
    st.header("2. Подбор оптимального показателя степени n")
    
    # Расширенный диапазон n
    n_min = 3.0
    n_max = 5.0
    n_step = 0.1
    n_candidates = np.arange(n_min, n_max + n_step, n_step)
    
    st.write(f"**Тестируем диапазон n от {n_min} до {n_max} с шагом {n_step}**")
    
    n_results = {}
    k_values_by_temp = {}
    
    # Анализ для каждого кандидата n
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, n in enumerate(n_candidates):
        status_text.text(f"Анализ для n = {n:.1f}...")
        
        k_values = []
        
        # Для каждой температуры в данных зерна №10
        for temp in df_grain10['T'].unique():
            temp_data = df_grain10[df_grain10['T'] == temp]
            
            if len(temp_data) >= 2:
                # Вычисляем d^n - d₀^n
                d_transformed = temp_data['d']**n - initial_diameter**n
                
                # Линейная регрессия: (d^n - d₀^n) = K * t
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        temp_data['t'], d_transformed
                    )
                    
                    # Сохраняем только физически осмысленные положительные K
                    if slope > 0:
                        k_values.append({
                            'T': temp,
                            'T_K': temp + 273.15,
                            'K': slope,
                            'R2': r_value**2,
                            'std_err': std_err,
                            'n_points': len(temp_data)
                        })
                except:
                    continue
        
        if k_values:
            k_df = pd.DataFrame(k_values)
            n_results[n] = {
                'k_df': k_df,
                'mean_R2': k_df['R2'].mean(),
                'min_R2': k_df['R2'].min(),
                'n_temperatures': len(k_df)
            }
            
            # Сохраняем K значения для лучшей визуализации
            if len(k_df) == len(df_grain10['T'].unique()):
                k_values_by_temp[n] = k_df.set_index('T')['K'].to_dict()
        
        progress_bar.progress((idx + 1) / len(n_candidates))
    
    status_text.text("✅ Анализ завершен!")
    
    # Анализ результатов подбора n
    if n_results:
        st.subheader("📊 Результаты подбора показателя n")
        
        # Создаем таблицу сравнения
        comparison_data = []
        for n, results in n_results.items():
            comparison_data.append({
                'n': n,
                'Средний R²': results['mean_R2'],
                'Минимальный R²': results['min_R2'],
                'Количество температур': results['n_temperatures']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Находим лучший n
        best_n_row = comparison_df.loc[comparison_df['Средний R²'].idxmax()]
        best_n = best_n_row['n']
        
        # Визуализация зависимости R² от n
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: R² vs n
        n_values = [data['n'] for data in comparison_data]
        mean_r2_values = [data['Средний R²'] for data in comparison_data]
        
        ax1.plot(n_values, mean_r2_values, 'b-o', linewidth=2, markersize=4)
        ax1.axvline(best_n, color='red', linestyle='--', 
                   label=f'Оптимальное n = {best_n:.1f}')
        ax1.set_xlabel('Показатель степени n')
        ax1.set_ylabel('Средний R²')
        ax1.set_title('Зависимость качества модели от показателя n')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Значения K для разных n (тепловая карта)
        if k_values_by_temp:
            # Создаем матрицу для тепловой карты
            temps = sorted(df_grain10['T'].unique())
            n_selected = [n for n in n_candidates if n in k_values_by_temp]
            
            k_matrix = []
            for n in n_selected:
                row = [k_values_by_temp[n].get(temp, np.nan) for temp in temps]
                k_matrix.append(row)
            
            k_matrix = np.array(k_matrix)
            
            im = ax2.imshow(k_matrix, cmap='viridis', aspect='auto', 
                           extent=[temps[0], temps[-1], n_selected[-1], n_selected[0]])
            ax2.set_xlabel('Температура (°C)')
            ax2.set_ylabel('Показатель n')
            ax2.set_title('Значения K в зависимости от n и температуры')
            plt.colorbar(im, ax=ax2, label='K')
            
            # Добавляем подписи
            for i in range(len(n_selected)):
                for j in range(len(temps)):
                    if not np.isnan(k_matrix[i, j]):
                        ax2.text(temps[j], n_selected[i], f'{k_matrix[i, j]:.3f}', 
                               ha='center', va='center', fontsize=8, color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Таблица сравнения с подсветкой лучшего
        def highlight_best(s):
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]
        
        st.dataframe(comparison_df.style.format({
            'Средний R²': '{:.4f}',
            'Минимальный R²': '{:.4f}'
        }).apply(highlight_best, subset=['Средний R²']))
        
        st.success(f"🎯 **Оптимальный показатель степени: n = {best_n:.1f}**")
        st.info(f"*Средний R² = {best_n_row['Средний R²']:.4f} для {best_n_row['Количество температур']} температур*")
        
        # Детальный анализ для лучшего n
        st.header(f"3. Детальный анализ для n = {best_n:.1f}")
        
        best_results = n_results[best_n]
        best_k_df = best_results['k_df']
        
        # Визуализация линейности для лучшего n
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        temps_to_plot = sorted(df_grain10['T'].unique())[:4]  # Первые 4 температуры
        
        for i, temp in enumerate(temps_to_plot):
            if i < len(axes):
                temp_data = df_grain10[df_grain10['T'] == temp]
                d_transformed = temp_data['d']**best_n - initial_diameter**best_n
                
                # Находим параметры регрессии
                k_value = best_k_df[best_k_df['T'] == temp]['K'].iloc[0]
                r2_value = best_k_df[best_k_df['T'] == temp]['R2'].iloc[0]
                
                axes[i].scatter(temp_data['t'], d_transformed, alpha=0.7, s=60)
                
                # Линия регрессии
                t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max(), 100)
                d_fit = k_value * t_range
                axes[i].plot(t_range, d_fit, 'r--', linewidth=2, 
                           label=f'K = {k_value:.4f}\nR² = {r2_value:.4f}')
                
                axes[i].set_xlabel('Время (часы)')
                axes[i].set_ylabel(f'$d^{{{best_n:.1f}}} - d_0^{{{best_n:.1f}}}$')
                axes[i].set_title(f'Температура {temp}°C')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Скрываем лишние subplots
        for i in range(len(temps_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Анализ Аррениуса для лучшего n
        st.header("4. Анализ Аррениуса")
        
        if len(best_k_df) >= 2:
            # Линейная регрессия: ln(K) = ln(K₀) - (Q/R) * (1/T)
            x = 1 / best_k_df['T_K']
            y = np.log(best_k_df['K'])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            Q = -slope * R  # Энергия активации в Дж/моль
            K0 = np.exp(intercept)
            
            # Визуализация Аррениуса
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # График 1: Аррениус
            ax1.scatter(x, y, s=100, color='blue', alpha=0.7, 
                       label='Экспериментальные точки')
            
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax1.plot(x_fit, y_fit, 'r--', linewidth=2,
                   label=f'Регрессия: Q = {Q:.0f} Дж/моль\nR² = {r_value**2:.4f}')
            
            ax1.set_xlabel('1/T (1/K)')
            ax1.set_ylabel('ln(K)')
            ax1.set_title('Уравнение Аррениуса для зерна №10')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Сравнение эксперимента и модели
            # Предсказанные значения диаметра
            ax2.scatter(df_grain10['t'], df_grain10['d'], alpha=0.7, s=60, 
                       label='Эксперимент', color='blue')
            
            # Предсказания модели для каждой температуры
            colors = plt.cm.viridis(np.linspace(0, 1, len(best_k_df)))
            for idx, (_, row) in enumerate(best_k_df.iterrows()):
                temp = row['T']
                k_pred = row['K']
                temp_data = df_grain10[df_grain10['T'] == temp]
                
                t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max(), 100)
                d_pred = (k_pred * t_range + initial_diameter**best_n)**(1/best_n)
                
                ax2.plot(t_range, d_pred, color=colors[idx], 
                        label=f'{temp}°C (модель)', linestyle='--')
            
            ax2.set_xlabel('Время (часы)')
            ax2.set_ylabel('Диаметр σ-фазы (мкм)')
            ax2.set_title('Сравнение эксперимента и модели')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Вывод параметров
            st.success("**Результаты анализа Аррениуса:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Энергия активации Q", f"{Q:.0f} Дж/моль")
            with col2:
                st.metric("Предэкспонента K₀", f"{K0:.6f}")
            with col3:
                st.metric("R²", f"{r_value**2:.4f}")
            with col4:
                st.metric("Показатель n", f"{best_n:.1f}")
            
            # Сохраняем финальные параметры
            st.session_state['final_params_grain10'] = {
                'n': best_n,
                'Q': Q,
                'K0': K0,
                'd0': initial_diameter,
                'arrhenius_R2': r_value**2
            }
            
            # Таблица с детальными результатами
            st.subheader("📋 Детальные результаты")
            results_df = best_k_df[['T', 'K', 'R2', 'n_points']].copy()
            results_df['K'] = results_df['K'].apply(lambda x: f"{x:.6f}")
            results_df['R2'] = results_df['R2'].apply(lambda x: f"{x:.4f}")
            st.dataframe(results_df)
            
            # Обратный расчет температуры
            st.header("5. Обратный расчет температуры")
            
            st.markdown(f"""
            **Формула для обратного расчета:**
            $$
            T = \\frac{{Q}}{{R \\cdot (\\ln K_0 - \\ln\\left(\\frac{{d^{{{best_n:.1f}}} - d_0^{{{best_n:.1f}}}}{{t}}\\right))}}
            $$
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                d_obs = st.number_input("Наблюдаемый диаметр d (мкм)", 
                                      value=5.0, min_value=0.1, step=0.1)
            with col2:
                t_obs = st.number_input("Время эксплуатации t (часы)", 
                                      value=5000, min_value=1, step=100)
            
            if st.button("Рассчитать температуру"):
                params = st.session_state['final_params_grain10']
                
                # Вычисляем K_obs
                k_obs = (d_obs**params['n'] - params['d0']**params['n']) / t_obs
                
                if k_obs > 0:
                    # Вычисляем температуру
                    denominator = R * (np.log(params['K0']) - np.log(k_obs))
                    if denominator > 0:
                        T_K = params['Q'] / denominator
                        T_C = T_K - 273.15
                        
                        st.success(f"**Расчетная температура эксплуатации: {T_C:.1f}°C**")
                        
                        # Детали расчета
                        st.write("**Детали расчета:**")
                        st.write(f"- K_obs = {k_obs:.6f}")
                        st.write(f"- ln(K_obs) = {np.log(k_obs):.4f}")
                        st.write(f"- ln(K₀) = {np.log(params['K0']):.4f}")
                        st.write(f"- Знаменатель = {denominator:.4f}")
                    else:
                        st.error("Ошибка расчета: отрицательный знаменатель")
                else:
                    st.error("Ошибка: K_obs должен быть положительным")
            
            # Выгрузка результатов
            st.header("6. Выгрузка результатов")
            
            if st.button("📊 Сгенерировать отчет для зерна №10"):
                output_buffer = io.BytesIO()
                
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    # Исходные данные
                    df_grain10.to_excel(writer, sheet_name='Исходные_данные', index=False)
                    
                    # Результаты подбора n
                    comparison_df.to_excel(writer, sheet_name='Подбор_показателя_n', index=False)
                    
                    # Детальные результаты для лучшего n
                    best_k_df.to_excel(writer, sheet_name='Кинетические_коэффициенты', index=False)
                    
                    # Финальные параметры
                    final_params_df = pd.DataFrame([st.session_state['final_params_grain10']])
                    final_params_df.to_excel(writer, sheet_name='Финальные_параметры', index=False)
                
                output_buffer.seek(0)
                
                st.download_button(
                    label="💾 Скачать отчет для зерна №10",
                    data=output_buffer,
                    file_name="отчет_зерно_10.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        else:
            st.warning("Для анализа Аррениуса нужно как минимум 2 температуры с данными")

# Информация о модели
with st.expander("📚 Информация о модели для зерна №10"):
    st.markdown("""
    **Упрощенная физическая модель для одного номера зерна:**
    
    1. **Степенной закон роста:**
       $$
       d^n - d_0^n = K \\cdot t
       $$
    
    2. **Температурная зависимость (Аррениус):**
       $$
       K = K_0 \\cdot \\exp\\left(-\\frac{Q}{RT}\\right)
       $$
    
    **Для зерна №10 определяем:**
    - Оптимальный показатель n (3.0-5.0)
    - Энергию активации Q
    - Предэкспоненту K₀
    
    **Ожидаемые значения:**
    - n ≈ 4.0 для диффузии по границам зерен
    - Q ≈ 200-300 кДж/моль для сталей
    """)
