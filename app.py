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
""")

# Загрузка данных
st.header("1. Загрузка данных для зерна №10")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                 value=0.0, min_value=0.0, step=0.1,
                                 help="Если зарождение завершено до ваших времен, используйте близкое к 0")

target_grain = 10

# Функции для расчета метрик качества
def calculate_comprehensive_metrics(y_true, y_pred):
    """Расчет комплексных метрик качества модели"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'R²': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}
    
    metrics = {
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
    }
    return metrics

def plot_with_diagnostics(ax, t_exp, y_exp, y_pred, t_range=None, y_range=None, 
                         title="", xlabel="Время (часы)", ylabel="", 
                         model_name="Модель"):
    """Улучшенная визуализация с диагностикой"""
    if len(t_exp) == 0:
        ax.text(0.5, 0.5, 'Нет данных', transform=ax.transAxes, ha='center')
        return
    
    # Экспериментальные точки
    ax.scatter(t_exp, y_exp, alpha=0.8, s=60, label='Эксперимент', color='blue')
    
    # Предсказания модели
    if t_range is not None and y_range is not None and len(t_range) > 0:
        ax.plot(t_range, y_range, 'r--', linewidth=2, label=model_name)
    
    # Соединяем экспериментальные точки
    sorted_idx = np.argsort(t_exp)
    ax.plot(t_exp.iloc[sorted_idx], y_exp.iloc[sorted_idx], 'b:', alpha=0.5, label='Тренд эксперимента')
    
    # Показываем ошибки
    for i, (t_val, y_true, y_pred_val) in enumerate(zip(t_exp, y_exp, y_pred)):
        ax.plot([t_val, t_val], [y_true, y_pred_val], 'gray', alpha=0.3)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Добавляем метрики качества в заголовок
    metrics = calculate_comprehensive_metrics(y_exp, y_pred)
    ax.text(0.02, 0.98, f"R² = {metrics['R²']:.3f}\nRMSE = {metrics['RMSE']:.2f}", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
                col1, col2, col3 = st.columns(3)
                with col1:
                    unique_temps = df_grain10['T'].unique()
                    st.metric("Температуры", f"{len(unique_temps)} уровней")
                with col2:
                    st.metric("Всего точек", f"{len(df_grain10)}")
                with col3:
                    st.metric("Диапазон времени", f"{df_grain10['t'].min()}-{df_grain10['t'].max()} ч")
                
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
                d_transformed = temp_data['d']**n - initial_diameter**n
                
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        temp_data['t'], d_transformed
                    )
                    
                    if slope > 0:
                        # Расчет предсказанных значений для метрик
                        d_pred_transformed = slope * temp_data['t'] + intercept
                        d_pred = (d_pred_transformed + initial_diameter**n)**(1/n)
                        
                        metrics = calculate_comprehensive_metrics(temp_data['d'], d_pred)
                        
                        k_values.append({
                            'T': temp, 'T_K': temp + 273.15, 'K': slope,
                            'R2': r_value**2, 'std_err': std_err,
                            'n_points': len(temp_data), 'metrics': metrics
                        })
                        available_temperatures.add(temp)
                except Exception as e:
                    st.write(f"Ошибка для n={n}, T={temp}: {e}")
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
                n_cols = 2
                n_rows = (len(temps_with_data) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                elif n_cols == 1:
                    axes = [[ax] for ax in axes]
                
                for idx, temp in enumerate(temps_with_data):
                    if idx < n_rows * n_cols:
                        row = idx // n_cols
                        col = idx % n_cols
                        
                        # Безопасное получение ax
                        if n_rows == 1:
                            ax = axes[col] if isinstance(axes, list) else axes
                        else:
                            ax = axes[row][col] if isinstance(axes[0], list) else axes[row, col]
                        
                        temp_data = df_grain10[df_grain10['T'] == temp]
                        
                        # Безопасное получение k_value
                        temp_k_data = best_k_df[best_k_df['T'] == temp]
                        if len(temp_k_data) > 0:
                            k_value = temp_k_data['K'].iloc[0]
                            
                            # Расчет предсказаний
                            t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max() * 1.2, 100)
                            d_pred_range = (k_value * t_range + initial_diameter**best_n)**(1/best_n)
                            
                            d_pred_points = (k_value * temp_data['t'] + initial_diameter**best_n)**(1/best_n)
                            
                            plot_with_diagnostics(
                                ax, temp_data['t'], temp_data['d'], d_pred_points,
                                t_range, d_pred_range, 
                                title=f'Температура {temp}°C',
                                ylabel='Диаметр (мкм)',
                                model_name=f'Модель (n={best_n:.1f})'
                            )
                        else:
                            ax.text(0.5, 0.5, f'Нет данных для {temp}°C', 
                                   transform=ax.transAxes, ha='center')
                            ax.set_title(f'Температура {temp}°C - нет данных')
                
                # Скрываем пустые subplots
                for idx in range(len(temps_with_data), n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    if n_rows == 1:
                        if isinstance(axes, list) and col < len(axes):
                            axes[col].set_visible(False)
                    else:
                        if (isinstance(axes, np.ndarray) and row < axes.shape[0] and col < axes.shape[1]):
                            axes[row, col].set_visible(False)
                        elif (isinstance(axes, list) and row < len(axes) and col < len(axes[row])):
                            axes[row][col].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Анализ расхождений
                st.subheader("Анализ расхождений модели и эксперимента")
                
                all_actual = []
                all_predicted = []
                all_temperatures = []
                
                for temp in temps_with_data:
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    temp_k_data = best_k_df[best_k_df['T'] == temp]
                    
                    if len(temp_k_data) > 0:
                        k_value = temp_k_data['K'].iloc[0]
                        d_pred = (k_value * temp_data['t'] + initial_diameter**best_n)**(1/best_n)
                        
                        all_actual.extend(temp_data['d'])
                        all_predicted.extend(d_pred)
                        all_temperatures.extend([temp] * len(temp_data))
                
                if len(all_actual) > 0:
                    # График остатков
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    residuals = np.array(all_actual) - np.array(all_predicted)
                    
                    # График 1: Остатки vs предсказания
                    ax1.scatter(all_predicted, residuals, alpha=0.7)
                    ax1.axhline(0, color='red', linestyle='--')
                    ax1.set_xlabel('Предсказанные значения диаметра (мкм)')
                    ax1.set_ylabel('Остатки (мкм)')
                    ax1.set_title('Остатки модели диаметров')
                    ax1.grid(True, alpha=0.3)
                    
                    # График 2: Фактические vs предсказанные значения
                    ax2.scatter(all_actual, all_predicted, alpha=0.7, c=all_temperatures, cmap='viridis')
                    min_val = min(min(all_actual), min(all_predicted))
                    max_val = max(max(all_actual), max(all_predicted))
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальное согласие')
                    ax2.set_xlabel('Фактические диаметры (мкм)')
                    ax2.set_ylabel('Предсказанные диаметры (мкм)')
                    ax2.set_title('Фактические vs предсказанные значения')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # Цветовая шкала для температур
                    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2, label='Температура (°C)')
                    
                    st.pyplot(fig)
                    
                    # Интерпретация результатов
                    overall_metrics = calculate_comprehensive_metrics(np.array(all_actual), np.array(all_predicted))
                    st.info(f"""
                    **Общая статистика модели диаметров:**
                    - Средний R²: {overall_metrics['R²']:.3f}
                    - Средняя ошибка (RMSE): {overall_metrics['RMSE']:.2f} мкм
                    - Средняя абсолютная ошибка: {overall_metrics['MAE']:.2f} мкм
                    
                    **Интерпретация:**
                    {'✅ Хорошее согласие' if overall_metrics['R²'] > 0.9 else '⚠️ Умеренное согласие' if overall_metrics['R²'] > 0.8 else '❌ Требуется улучшение модели'}
                    """)
            
            # Анализ Аррениуса для диаметров
            if len(best_k_df) >= 2:
                st.subheader("Аррениус-анализ для диаметров")
                
                x = 1 / best_k_df['T_K']
                y = np.log(best_k_df['K'])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                Q_diam = -slope * R
                K0_diam = np.exp(intercept)
                
                # Визуализация Аррениуса
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(x, y, s=100, alpha=0.7, label='Экспериментальные точки')
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2,
                       label=f'Регрессия: Q = {Q_diam:.0f} Дж/моль\nR² = {r_value**2:.4f}')
                
                ax.set_xlabel('1/T (1/K)')
                ax.set_ylabel('ln(K)')
                ax.set_title('Аррениус для кинетических коэффициентов диаметров')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.session_state['diam_params'] = {
                    'n': best_n, 'Q': Q_diam, 'K0': K0_diam, 'd0': initial_diameter,
                    'arrhenius_R2': r_value**2
                }
        else:
            st.error("❌ Не удалось подобрать параметры модели для любого n")

    st.header("3. 📊 Анализ содержания σ-фазы (JMAK-модель)")
    
    with st.expander("💡 Объяснение JMAK-анализа"):
        st.markdown("""
        **Что анализируем:** Кинетику фазового превращения через долю σ-фазы
        
        **JMAK-модель:** 
        $$ X(t) = X_\\infty \\cdot [1 - \\exp(-(k \\cdot t)^m)] $$
        
        **Параметры:**
        - k: скорость превращения
        - m: показатель, связанный с механизмом зарождения  
        - X∞: предельное содержание фазы
        
        **Ожидаемое поведение:**
        - S-образная кривая роста
        - Насыщение при больших временах
        - R² должен быть близок к 1
        """)
    
    # JMAK-анализ
    jmak_results = {}
    
    for temp in df_grain10['T'].unique():
        temp_data = df_grain10[df_grain10['T'] == temp]
        
        if len(temp_data) >= 3:
            try:
                # Более надежный подбор параметров
                bounds = ([1e-6, 0.1, 50], [1.0, 3.0, 150])
                p0 = [0.001, 1.0, 100]
                
                popt, pcov = curve_fit(
                    lambda t, k, m, X_inf: X_inf * (1 - np.exp(-(k * t) ** m)),
                    temp_data['t'], temp_data['f'],
                    p0=p0, bounds=bounds, maxfev=5000
                )
                
                k_jmak, m_jmak, X_inf = popt
                y_pred = X_inf * (1 - np.exp(-(k_jmak * temp_data['t']) ** m_jmak))
                metrics = calculate_comprehensive_metrics(temp_data['f'], y_pred)
                
                jmak_results[temp] = {
                    'k': k_jmak, 'm': m_jmak, 'X_inf': X_inf,
                    'R2': metrics['R²'], 'metrics': metrics
                }
                
            except Exception as e:
                st.write(f"⚠️ Не удалось подобрать JMAK для {temp}°C: {str(e)[:100]}...")
    
    if jmak_results:
        # Визуализация JMAK для всех температур
        st.subheader("Диагностика JMAK-модели по температурам")
        
        temps_jmak = sorted(jmak_results.keys())
        
        if len(temps_jmak) > 0:
            n_cols_jmak = 2
            n_rows_jmak = (len(temps_jmak) + n_cols_jmak - 1) // n_cols_jmak
            
            fig, axes = plt.subplots(n_rows_jmak, n_cols_jmak, figsize=(15, 5*n_rows_jmak))
            if n_rows_jmak == 1:
                axes = [axes] if n_cols_jmak == 1 else axes
            
            for idx, temp in enumerate(temps_jmak):
                if idx < n_rows_jmak * n_cols_jmak:
                    row = idx // n_cols_jmak
                    col = idx % n_cols_jmak
                    
                    if n_rows_jmak == 1:
                        ax = axes[col] if isinstance(axes, list) else axes
                    else:
                        if isinstance(axes, np.ndarray):
                            ax = axes[row, col]
                        else:
                            ax = axes[row][col]
                    
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    results = jmak_results[temp]
                    
                    # Расчет JMAK-кривой
                    t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max() * 1.2, 100)
                    y_pred_range = results['X_inf'] * (1 - np.exp(-(results['k'] * t_range) ** results['m']))
                    
                    y_pred_points = results['X_inf'] * (1 - np.exp(-(results['k'] * temp_data['t']) ** results['m']))
                    
                    plot_with_diagnostics(
                        ax, temp_data['t'], temp_data['f'], y_pred_points,
                        t_range, y_pred_range,
                        title=f'Температура {temp}°C',
                        ylabel='Содержание σ-фазы (%)',
                        model_name=f'JMAK (m={results["m"]:.2f})'
                    )
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Анализ качества JMAK
            st.subheader("Сводка по JMAK-анализу")
            
            jmak_summary = []
            for temp, results in jmak_results.items():
                jmak_summary.append({
                    'Температура, °C': temp,
                    'k': f"{results['k']:.6f}",
                    'm': f"{results['m']:.2f}",
                    'X∞': f"{results['X_inf']:.1f}%",
                    'R²': f"{results['R2']:.3f}",
                    'RMSE': f"{results['metrics']['RMSE']:.2f}"
                })
            
            st.table(pd.DataFrame(jmak_summary))
        else:
            st.warning("❌ Нет температур с успешным JMAK-подбором")

    st.header("4. 🎯 Обратный расчет температуры")
    
    if 'diam_params' in st.session_state:
        params = st.session_state['diam_params']
        
        col1, col2 = st.columns(2)
        with col1:
            d_obs = st.number_input("Наблюдаемый диаметр d (мкм)", value=5.0, min_value=0.1, step=0.1)
        with col2:
            t_obs = st.number_input("Время эксплуатации t (часы)", value=5000, min_value=1, step=100)
        
        if st.button("Рассчитать температуру по диаметру"):
            try:
                k_obs = (d_obs**params['n'] - params['d0']**params['n']) / t_obs
                
                if k_obs > 0:
                    denominator = R * (np.log(params['K0']) - np.log(k_obs))
                    if denominator > 0:
                        T_K = params['Q'] / denominator
                        T_C = T_K - 273.15
                        
                        st.success(f"**Расчетная температура: {T_C:.1f}°C**")
                        
                        # Оценка надежности
                        reliability = "✅ Надежность хорошая" if params['arrhenius_R2'] > 0.9 else "⚠️ Надежность умеренная" if params['arrhenius_R2'] > 0.8 else "❌ Надежность низкая"
                        
                        st.info(f"""
                        **Оценка надежности расчета:**
                        - R² Аррениуса: {params['arrhenius_R2']:.3f}
                        - Энергия активации: {params['Q']:.0f} Дж/моль
                        - Показатель n: {params['n']:.1f}
                        
                        {reliability}
                        """)
                    else:
                        st.error("❌ Ошибка: отрицательный знаменатель в расчете температуры")
                else:
                    st.error("❌ Ошибка: K_obs должен быть положительным")
            except Exception as e:
                st.error(f"❌ Ошибка расчета: {e}")
    else:
        st.warning("❌ Сначала выполните анализ диаметров для калибровки модели")
