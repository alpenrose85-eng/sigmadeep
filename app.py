import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
import io
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Универсальная газовая постоянная
R = 8.314  # Дж/(моль·К)

st.title("🔬 Комплексный анализ кинетики σ-фазы с оценкой качества моделей")
st.markdown("""
**Улучшенный анализ с диагностикой качества подбора:**
- Полная оценка ошибок для всех температур
- Детальная диагностика расхождений модели и эксперимента
- Объяснения всех графиков и метрик
- **НОВОЕ: Анализ содержания σ-фазы по JMAK-модели**
""")

# Загрузка данных
st.header("1. Загрузка данных для зерна №10")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
col1, col2 = st.columns(2)
with col1:
    initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                     value=0.1, min_value=0.0, step=0.1,
                                     help="Рекомендуется использовать небольшое положительное значение (0.1-0.5 мкм)")
with col2:
    enable_phase_analysis = st.checkbox("Включить анализ содержания фазы (JMAK)", 
                                      value=True, 
                                      help="Анализ кинетики фазового превращения по содержанию σ-фазы")

target_grain = 10

# Функции для расчета метрик качества
def calculate_comprehensive_metrics(y_true, y_pred):
    """Расчет комплексных метрик качества модели"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'R²': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}
    
    # Убедимся, что массивы имеют одинаковую длину
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    try:
        metrics = {
            'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
        }
        return metrics
    except:
        return {'R²': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}

def safe_plot_with_diagnostics(ax, t_exp, y_exp, y_pred, t_range=None, y_range=None, 
                              title="", xlabel="Время (часы)", ylabel="", 
                              model_name="Модель"):
    """Безопасная визуализация с диагностикой"""
    try:
        # Очищаем оси
        ax.clear()
        
        if len(t_exp) == 0 or len(y_exp) == 0:
            ax.text(0.5, 0.5, 'Нет данных', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return
        
        # Проверяем данные на валидность
        valid_mask = ~np.isnan(t_exp) & ~np.isnan(y_exp) & ~np.isnan(y_pred)
        t_exp = t_exp[valid_mask]
        y_exp = y_exp[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(t_exp) == 0:
            ax.text(0.5, 0.5, 'Нет валидных данных', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return
        
        # Экспериментальные точки
        ax.scatter(t_exp, y_exp, alpha=0.8, s=60, label='Эксперимент', color='blue')
        
        # Предсказания модели (линия)
        if t_range is not None and y_range is not None and len(t_range) > 0 and len(y_range) > 0:
            # Проверяем, что предсказания физически осмысленны (положительные диаметры)
            if ylabel == 'Диаметр (мкм)':
                valid_range_mask = y_range > 0
                if np.any(valid_range_mask):
                    ax.plot(t_range[valid_range_mask], y_range[valid_range_mask], 'r--', 
                           linewidth=2, label=model_name)
            else:
                ax.plot(t_range, y_range, 'r--', linewidth=2, label=model_name)
        
        # Соединяем экспериментальные точки
        sorted_idx = np.argsort(t_exp)
        ax.plot(t_exp.iloc[sorted_idx] if hasattr(t_exp, 'iloc') else t_exp[sorted_idx], 
               y_exp.iloc[sorted_idx] if hasattr(y_exp, 'iloc') else y_exp[sorted_idx], 
               'b:', alpha=0.5, label='Тренд эксперимента')
        
        # Показываем ошибки (вертикальные линии)
        for i in range(min(len(t_exp), len(y_exp), len(y_pred))):
            t_val = t_exp.iloc[i] if hasattr(t_exp, 'iloc') else t_exp[i]
            y_true = y_exp.iloc[i] if hasattr(y_exp, 'iloc') else y_exp[i]
            y_pred_val = y_pred.iloc[i] if hasattr(y_pred, 'iloc') else y_pred[i]
            ax.plot([t_val, t_val], [y_true, y_pred_val], 'gray', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем метрики качества
        metrics = calculate_comprehensive_metrics(y_exp, y_pred)
        ax.text(0.02, 0.98, f"R² = {metrics['R²']:.3f}\nRMSE = {metrics['RMSE']:.2f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка построения:\n{str(e)[:50]}...', 
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_title(title)

# Функции для JMAK-анализа
def jmak_model(t, k, n):
    """JMAK модель: X(t) = 1 - exp(-(k*t)^n)"""
    return 1 - np.exp(-(k * t) ** n)

def fit_jmak_model(time, f_phase, initial_n=1.0):
    """Подбор параметров JMAK модели"""
    # Преобразуем проценты в доли (0-1)
    f_normalized = np.array(f_phase) / 100.0
    
    # Проверяем, что данные валидны
    valid_mask = ~np.isnan(time) & ~np.isnan(f_normalized) & (f_normalized >= 0) & (f_normalized <= 1)
    time_valid = time[valid_mask]
    f_valid = f_normalized[valid_mask]
    
    if len(time_valid) < 2:
        return None, None, None
    
    try:
        # Начальные приближения
        k_guess = 1.0 / np.mean(time_valid) if np.mean(time_valid) > 0 else 0.1
        
        # Подгонка модели
        popt, pcov = curve_fit(jmak_model, time_valid, f_valid, 
                              p0=[k_guess, initial_n],
                              bounds=([1e-6, 0.1], [10, 4]),
                              maxfev=5000)
        
        k_fit, n_fit = popt
        return k_fit, n_fit, pcov
    
    except Exception as e:
        st.warning(f"Ошибка подбора JMAK для температуры: {e}")
        return None, None, None

def calculate_jmak_predictions(time, k, n):
    """Расчет предсказаний JMAK модели"""
    return jmak_model(time, k, n) * 100  # Возвращаем в процентах

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        required_cols = ['G', 'T', 't', 'd', 'f']
        
        if all(col in df.columns for col in required_cols):
            df_grain10 = df[df['G'] == target_grain].copy()
            
            if len(df_grain10) > 0:
                st.session_state['grain10_data'] = df_grain10
                st.success(f"✅ Данные для зерна №10 успешно загружены!")
                
                # Статистика
                st.subheader("📊 Статистика данных для зерна №10:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    unique_temps = df_grain10['T'].unique()
                    st.metric("Температуры", f"{len(unique_temps)} уровней")
                with col2:
                    st.metric("Всего точек", f"{len(df_grain10)}")
                with col3:
                    st.metric("Диапазон времени", f"{df_grain10['t'].min()}-{df_grain10['t'].max()} ч")
                with col4:
                    st.metric("Содержание фазы", f"{df_grain10['f'].min():.1f}-{df_grain10['f'].max():.1f}%")
                
                # Проверка данных на аномалии
                st.subheader("🔍 Проверка данных на аномалии")
                
                if (df_grain10['d'] <= 0).any():
                    st.warning(f"⚠️ Обнаружены {sum(df_grain10['d'] <= 0)} точек с отрицательными или нулевыми диаметрами")
                    st.write("Это физически невозможно. Проверьте исходные данные.")
                
                if (df_grain10['f'] < 0).any() or (df_grain10['f'] > 100).any():
                    st.warning(f"⚠️ Обнаружены точки с содержанием фазы вне диапазона 0-100%")
                
                st.dataframe(df_grain10.head(10))
                
            else:
                st.error(f"❌ В данных нет записей для зерна №10")
        else:
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"❌ Отсутствуют колонки: {missing}")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

# Основной расчет
if 'grain10_data' in st.session_state:
    df_grain10 = st.session_state['grain10_data']
    
    # Фильтрация аномальных данных
    df_grain10_clean = df_grain10[(df_grain10['d'] > 0) & (df_grain10['f'] >= 0) & (df_grain10['f'] <= 100)].copy()
    
    if len(df_grain10_clean) < len(df_grain10):
        st.warning(f"⚠️ Удалено {len(df_grain10) - len(df_grain10_clean)} аномальных точек")
        df_grain10 = df_grain10_clean
    
    df_grain10['T_K'] = df_grain10['T'] + 273.15
    
    st.header("2. 📏 Анализ диаметров σ-фазы")
    
    with st.expander("💡 Объяснение анализа диаметров"):
        st.markdown("""
        **Что анализируем:** Рост среднего диаметра частиц σ-фазы во времени
        
        **Физическая модель:** 
        $$ d^n - d_0^n = K \\cdot t $$
        
        **Ожидаемое поведение:**
        - При правильном n график $d^n - d_0^n$ vs t должен быть линейным
        - Качество оценивается по R² близкому к 1
        - Остатки должны быть случайными (без тренда)
        
        **Как оценивать результат:**
        - R² > 0.95 - отличное согласие
        - R² 0.90-0.95 - хорошее согласие  
        - R² < 0.90 - требуется улучшение модели
        """)
    
    # Подбор показателя степени n
    st.subheader("Поиск оптимального показателя степени n")
    
    n_min, n_max, n_step = 3.0, 5.0, 0.1
    n_candidates = np.arange(n_min, n_max + n_step, n_step)
    
    n_results = {}
    available_temperatures = set()
    
    for n in n_candidates:
        k_values = []
        
        for temp in df_grain10['T'].unique():
            temp_data = df_grain10[df_grain10['T'] == temp]
            
            if len(temp_data) >= 2:
                # Проверяем, что преобразование не дает отрицательных значений
                d_transformed = temp_data['d']**n - initial_diameter**n
                
                # Если есть отрицательные значения - пропускаем эту температуру для данного n
                if (d_transformed < 0).any():
                    continue
                
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        temp_data['t'], d_transformed
                    )
                    
                    if slope > 0:
                        # Расчет предсказанных значений
                        d_pred_transformed = slope * temp_data['t'] + intercept
                        d_pred = (d_pred_transformed + initial_diameter**n)**(1/n)
                        
                        # Проверяем, что предсказанные диаметры положительные
                        if (d_pred > 0).all():
                            metrics = calculate_comprehensive_metrics(temp_data['d'].values, d_pred)
                            
                            k_values.append({
                                'T': temp, 'T_K': temp + 273.15, 'K': slope,
                                'R2': r_value**2, 'std_err': std_err,
                                'n_points': len(temp_data), 'metrics': metrics
                            })
                            available_temperatures.add(temp)
                except Exception as e:
                    continue
        
        if k_values:
            k_df = pd.DataFrame(k_values)
            overall_r2 = k_df['R2'].mean()
            n_results[n] = {
                'k_df': k_df, 'mean_R2': overall_r2,
                'min_R2': k_df['R2'].min(), 'n_temperatures': len(k_df)
            }
    
    # Визуализация подбора n
    if n_results:
        comparison_data = []
        for n, results in n_results.items():
            comparison_data.append({
                'n': n, 'Средний R²': results['mean_R2'],
                'Минимальный R²': results['min_R2'], 
                'Количество температур': results['n_temperatures']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) > 0:
            best_n_row = comparison_df.loc[comparison_df['Средний R²'].idxmax()]
            best_n = best_n_row['n']
            
            st.success(f"🎯 Оптимальный показатель: n = {best_n:.1f} (R² = {best_n_row['Средний R²']:.3f})")
            
            # ДИАГНОСТИКА КАЧЕСТВА ПОДБОРА ДЛЯ ЛУЧШЕГО n
            st.subheader(f"Диагностика качества модели для n = {best_n:.1f}")
            
            best_k_df = n_results[best_n]['k_df']
            
            # Визуализация для всех температур с данными
            temps_with_data = sorted(available_temperatures)
            
            if len(temps_with_data) > 0:
                n_cols = min(2, len(temps_with_data))
                n_rows = (len(temps_with_data) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                
                # Делаем axes всегда двумерным массивом для единообразия
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = np.array([axes])
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for idx, temp in enumerate(temps_with_data):
                    if idx < n_rows * n_cols:
                        row = idx // n_cols
                        col = idx % n_cols
                        
                        ax = axes[row, col]
                        temp_data = df_grain10[df_grain10['T'] == temp]
                        
                        # Безопасное получение k_value
                        temp_k_data = best_k_df[best_k_df['T'] == temp]
                        if len(temp_k_data) > 0:
                            k_value = temp_k_data['K'].iloc[0]
                            
                            # Расчет предсказаний
                            t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max() * 1.2, 100)
                            d_pred_range = (k_value * t_range + initial_diameter**best_n)**(1/best_n)
                            
                            # Убедимся, что предсказания положительные
                            d_pred_range = np.maximum(d_pred_range, 0.1)  # Минимальный диаметр 0.1 мкм
                            
                            d_pred_points = (k_value * temp_data['t'] + initial_diameter**best_n)**(1/best_n)
                            d_pred_points = np.maximum(d_pred_points, 0.1)
                            
                            safe_plot_with_diagnostics(
                                ax, temp_data['t'].values, temp_data['d'].values, d_pred_points,
                                t_range, d_pred_range, 
                                title=f'Температура {temp}°C',
                                ylabel='Диаметр (мкм)',
                                model_name=f'Модель (n={best_n:.1f})'
                            )
                        else:
                            ax.text(0.5, 0.5, f'Нет данных для {temp}°C', 
                                   transform=ax.transAxes, ha='center', va='center')
                            ax.set_title(f'Температура {temp}°C')
                
                # Скрываем пустые subplots
                for idx in range(len(temps_with_data), n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Анализ расхождений модели и эксперимента
                st.subheader("📊 Анализ расхождений модели и эксперимента")
                
                with st.expander("💡 Объяснение графиков расхождений"):
                    st.markdown("""
                    **График остатков (Residuals Plot):**
                    - Показывает разницу между экспериментальными и предсказанными значениями
                    - **Идеально:** точки случайно разбросаны вокруг нулевой линии
                    - **Проблема:** видимый тренд или структура в остатках
                    
                    **График фактические vs предсказанные:**
                    - Показывает общее качество предсказаний
                    - **Идеально:** точки близко к диагональной линии
                    - Цвет точек показывает температуру эксперимента
                    """)
                
                all_actual = []
                all_predicted = []
                all_temperatures = []
                
                for temp in temps_with_data:
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    temp_k_data = best_k_df[best_k_df['T'] == temp]
                    
                    if len(temp_k_data) > 0:
                        k_value = temp_k_data['K'].iloc[0]
                        d_pred = (k_value * temp_data['t'] + initial_diameter**best_n)**(1/best_n)
                        d_pred = np.maximum(d_pred, 0.1)  # Защита от отрицательных значений
                        
                        all_actual.extend(temp_data['d'].values)
                        all_predicted.extend(d_pred)
                        all_temperatures.extend([temp] * len(temp_data))
                
                if len(all_actual) > 0:
                    # График остатков
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    residuals = np.array(all_actual) - np.array(all_predicted)
                    
                    # График 1: Остатки vs предсказания
                    ax1.scatter(all_predicted, residuals, alpha=0.7)
                    ax1.axhline(0, color='red', linestyle='--', label='Нулевая ошибка')
                    ax1.set_xlabel('Предсказанные значения диаметра (мкм)')
                    ax1.set_ylabel('Остатки = Факт - Прогноз (мкм)')
                    ax1.set_title('Остатки модели диаметров\n(чем ближе к нулю - тем лучше)')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # График 2: Фактические vs предсказанные значения
                    scatter = ax2.scatter(all_actual, all_predicted, alpha=0.7, 
                                        c=all_temperatures, cmap='viridis', s=60)
                    min_val = min(min(all_actual), min(all_predicted))
                    max_val = max(max(all_actual), max(all_predicted))
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
                            linewidth=2, label='Идеальное согласие')
                    ax2.set_xlabel('Фактические диаметры (мкм)')
                    ax2.set_ylabel('Предсказанные диаметры (мкм)')
                    ax2.set_title('Фактические vs предсказанные значения\n(чем ближе к линии - тем лучше)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # Цветовая шкала для температур
                    cbar = plt.colorbar(scatter, ax=ax2)
                    cbar.set_label('Температура (°C)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Общая статистика
                    overall_metrics = calculate_comprehensive_metrics(np.array(all_actual), np.array(all_predicted))
                    st.info(f"""
                    **📈 Общая статистика модели диаметров:**
                    - **R² = {overall_metrics['R²']:.3f}** - доля объясненной дисперсии
                    - **RMSE = {overall_metrics['RMSE']:.2f} мкм** - средняя ошибка предсказания
                    - **MAE = {overall_metrics['MAE']:.2f} мкм** - средняя абсолютная ошибка
                    - **MAPE = {overall_metrics['MAPE']:.1f}%** - средняя процентная ошибка
                    
                    **🎯 Оценка качества:**
                    { '✅ Отличное согласие' if overall_metrics['R²'] > 0.95 else 
                      '🟡 Хорошее согласие' if overall_metrics['R²'] > 0.85 else 
                      '🟠 Умеренное согласие' if overall_metrics['R²'] > 0.7 else 
                      '🔴 Требуется улучшение модели'}
                    """)

    # НОВЫЙ РАЗДЕЛ: Анализ содержания σ-фазы по JMAK-модели
    if enable_phase_analysis:
        st.header("3. 📊 Анализ содержания σ-фазы (JMAK-модель)")
        
        with st.expander("💡 Объяснение JMAK-анализа"):
            st.markdown("""
            **Что анализируем:** Кинетику фазового превращения по изменению объемной доли σ-фазы
            
            **Физическая модель (Johnson-Mehl-Avrami-Kolmogorov):**
            $$ X(t) = 1 - \\exp\\left(-(k \\cdot t)^n\\right) $$
            
            **Параметры модели:**
            - **k** - константа скорости превращения
            - **n** - показатель Аврами (характеризует механизм превращения)
            
            **Типичные значения n:**
            - n ≈ 1 - насыщение центров зарождения
            - n ≈ 2-4 - зарождение и рост (диффузионный контроль)
            
            **Как оценивать результат:**
            - R² > 0.95 - отличное согласие
            - Остатки должны быть случайными
            - Модель должна хорошо описывать S-образную кривую
            """)
        
        # Подбор параметров JMAK для каждой температуры
        st.subheader("Подбор параметров JMAK-модели")
        
        jmak_results = {}
        all_phase_actual = []
        all_phase_predicted = []
        all_phase_temperatures = []
        
        for temp in df_grain10['T'].unique():
            temp_data = df_grain10[df_grain10['T'] == temp].copy()
            
            if len(temp_data) >= 3:  # Минимум 3 точки для подбора
                k_fit, n_fit, pcov = fit_jmak_model(temp_data['t'].values, temp_data['f'].values)
                
                if k_fit is not None and n_fit is not None:
                    # Расчет предсказаний
                    f_pred = calculate_jmak_predictions(temp_data['t'].values, k_fit, n_fit)
                    metrics = calculate_comprehensive_metrics(temp_data['f'].values, f_pred)
                    
                    jmak_results[temp] = {
                        'k': k_fit, 'n': n_fit, 'metrics': metrics,
                        'data': temp_data, 'predictions': f_pred
                    }
                    
                    # Собираем данные для общего анализа
                    all_phase_actual.extend(temp_data['f'].values)
                    all_phase_predicted.extend(f_pred)
                    all_phase_temperatures.extend([temp] * len(temp_data))
        
        # Визуализация результатов JMAK
        if jmak_results:
            # Создаем таблицу результатов
            results_data = []
            for temp, results in jmak_results.items():
                results_data.append({
                    'Температура (°C)': temp,
                    'k (ч⁻¹)': results['k'],
                    'n': results['n'],
                    'R²': results['metrics']['R²'],
                    'RMSE': results['metrics']['RMSE'],
                    'MAPE': results['metrics']['MAPE']
                })
            
            results_df = pd.DataFrame(results_data)
            st.subheader("Параметры JMAK-модели по температурам")
            st.dataframe(results_df.style.format({
                'k (ч⁻¹)': '{:.4f}',
                'n': '{:.3f}',
                'R²': '{:.3f}',
                'RMSE': '{:.2f}',
                'MAPE': '{:.1f}'
            }))
            
            # Графики для каждой температуры
            st.subheader("Визуализация JMAK-подбора по температурам")
            
            temps_jmak = sorted(jmak_results.keys())
            n_cols = min(2, len(temps_jmak))
            n_rows = (len(temps_jmak) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = np.array([axes])
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, temp in enumerate(temps_jmak):
                if idx < n_rows * n_cols:
                    row = idx // n_cols
                    col = idx % n_cols
                    
                    ax = axes[row, col]
                    results = jmak_results[temp]
                    temp_data = results['data']
                    
                    # Экспериментальные точки
                    ax.scatter(temp_data['t'], temp_data['f'], alpha=0.8, s=60, 
                              label='Эксперимент', color='blue')
                    
                    # JMAK кривая
                    t_range = np.linspace(0, temp_data['t'].max() * 1.2, 100)
                    f_range = calculate_jmak_predictions(t_range, results['k'], results['n'])
                    ax.plot(t_range, f_range, 'r--', linewidth=2, 
                           label=f'JMAK (k={results["k"]:.3f}, n={results["n"]:.2f})')
                    
                    ax.set_xlabel('Время (часы)')
                    ax.set_ylabel('Содержание фазы (%)')
                    ax.set_title(f'Температура {temp}°C\nR² = {results["metrics"]["R²"]:.3f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # Скрываем пустые subplots
            for idx in range(len(temps_jmak), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Анализ расхождений JMAK модели
            if len(all_phase_actual) > 0:
                st.subheader("📊 Анализ расхождений JMAK-модели")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                residuals = np.array(all_phase_actual) - np.array(all_phase_predicted)
                
                # График 1: Остатки vs предсказания
                ax1.scatter(all_phase_predicted, residuals, alpha=0.7)
                ax1.axhline(0, color='red', linestyle='--', label='Нулевая ошибка')
                ax1.set_xlabel('Предсказанные значения содержания фазы (%)')
                ax1.set_ylabel('Остатки = Факт - Прогноз (%)')
                ax1.set_title('Остатки JMAK-модели\n(чем ближе к нулю - тем лучше)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # График 2: Фактические vs предсказанные значения
                scatter = ax2.scatter(all_phase_actual, all_phase_predicted, alpha=0.7, 
                                    c=all_phase_temperatures, cmap='viridis', s=60)
                ax2.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Идеальное согласие')
                ax2.set_xlabel('Фактическое содержание фазы (%)')
                ax2.set_ylabel('Предсказанное содержание фазы (%)')
                ax2.set_title('Фактические vs предсказанные значения\nJMAK-модели')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Температура (°C)')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Общая статистика JMAK
                overall_phase_metrics = calculate_comprehensive_metrics(
                    np.array(all_phase_actual), np.array(all_phase_predicted)
                )
                
                st.info(f"""
                **📈 Общая статистика JMAK-модели:**
                - **R² = {overall_phase_metrics['R²']:.3f}** - доля объясненной дисперсии
                - **RMSE = {overall_phase_metrics['RMSE']:.2f}%** - средняя ошибка предсказания
                - **MAE = {overall_phase_metrics['MAE']:.2f}%** - средняя абсолютная ошибка
                - **MAPE = {overall_phase_metrics['MAPE']:.1f}%** - средняя процентная ошибка
                
                **🎯 Оценка качества:**
                { '✅ Отличное согласие' if overall_phase_metrics['R²'] > 0.95 else 
                  '🟡 Хорошее согласие' if overall_phase_metrics['R²'] > 0.85 else 
                  '🟠 Умеренное согласие' if overall_phase_metrics['R²'] > 0.7 else 
                  '🔴 Требуется улучшение модели'}
                """)
        
        else:
            st.warning("❌ Не удалось подобрать параметры JMAK-модели для доступных данных")

st.header("🎯 Рекомендации по улучшению модели")

st.markdown("""
**Если модель показывает плохое согласие (R² < 0.8):**

1. **Проверьте данные:**
   - Нет ли отрицательных диаметров
   - Корректны ли единицы измерения
   - Проверьте выбросы

2. **Настройте параметры:**
   - Попробуйте другой начальный диаметр d₀
   - Расширьте диапазон поиска n
   - Проверьте физическую осмысленность параметров

3. **Рассмотрите альтернативные модели:**
   - Более сложные степенные законы
   - Модифицированные JMAK-модели
   - Учет дополнительных факторов

**Отрицательные диаметры в остатках:** Это разница между экспериментом и моделью, а не реальные диаметры!
- Остаток = Фактический диаметр - Предсказанный диаметр
- Отрицательный остаток означает, что модель переоценила диаметр
- Положительный остаток - модель недооценила диаметр

**Для JMAK-анализа:**
- Убедитесь, что данные охватывают всю S-образную кривую
- Проверьте физическую осмысленность параметров n
- Рассмотрите возможность фиксации n для всех температур
""")
