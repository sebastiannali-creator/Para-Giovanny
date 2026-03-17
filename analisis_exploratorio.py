
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

class FatigueDatasetExplorer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.metadata = None
        self.participants = []
        self.session_mapping = {}
        self.data_summary = {}
        
    def load_metadata(self):
        print("=== ANÁLISIS DE METADATA ===")
        self.metadata = pd.read_csv(self.base_path / 'metadata.csv')
        print(f"Número total de participantes: {len(self.metadata)}")
        print(f"Columnas en metadata: {list(self.metadata.columns)}")
        print("\nPrimeras 5 filas de metadata:")
        print(self.metadata.head())
        
        for _, row in self.metadata.iterrows():
            participant_id = f"{int(row['participant_id']):02d}"
            self.participants.append(participant_id)
            self.session_mapping[participant_id] = {
                'low': f"{int(row['low_session']):02d}",
                'medium': f"{int(row['medium_session']):02d}", 
                'high': f"{int(row['high_session']):02d}"
            }
        
        print(f"\nParticipantes encontrados: {self.participants}")
        print(f"Ejemplo de mapping de sesiones para participante 01: {self.session_mapping['01']}")
        
    def explore_file_structure(self):
        print("\n=== ESTRUCTURA DE ARCHIVOS ===")
        
        file_types = set()
        total_files = 0
        
        for participant in self.participants:
            participant_path = self.base_path / participant
            if participant_path.exists():
                for session in ['01', '02', '03']:
                    session_path = participant_path / session
                    if session_path.exists():
                        for file_path in session_path.glob('*.csv'):
                            file_types.add(file_path.name)
                            total_files += 1
        
        self.file_types = sorted(list(file_types))
        print(f"Tipos de archivos encontrados ({len(self.file_types)}):")
        for i, file_type in enumerate(self.file_types, 1):
            print(f"{i:2d}. {file_type}")
        
        print(f"\nTotal de archivos CSV: {total_files}")
        
        self.categorize_files()
        
    def categorize_files(self):
        categories = {
            'Chest (ECG/Breathing)': [],
            'EEG (Forehead)': [],
            'Wrist (Physiological)': [],
            'Ear (PPG/Acc/Gyro)': [],
            'Experimental Tasks': [],
            'Device Status': [],
            'Other': []
        }
        
        for file_type in self.file_types:
            if file_type.startswith('chest_'):
                categories['Chest (ECG/Breathing)'].append(file_type)
            elif file_type.startswith('forehead_'):
                categories['EEG (Forehead)'].append(file_type)
            elif file_type.startswith('wrist_'):
                categories['Wrist (Physiological)'].append(file_type)
            elif file_type.startswith('ear_'):
                categories['Ear (PPG/Acc/Gyro)'].append(file_type)
            elif file_type.startswith('exp_'):
                categories['Experimental Tasks'].append(file_type)
            elif 'device' in file_type or 'status' in file_type or 'battery' in file_type:
                categories['Device Status'].append(file_type)
            else:
                categories['Other'].append(file_type)
        
        print("\n=== CATEGORIZACIÓN POR SENSOR ===")
        for category, files in categories.items():
            if files:
                print(f"\n{category} ({len(files)} archivos):")
                for file in files:
                    print(f"  - {file}")
        
        self.categories = categories
        
    def analyze_sample_data(self):
        print("\n=== ANÁLISIS DE DATOS DE MUESTRA ===")
        
        sample_participant = '01'
        sample_session = '01'
        sample_path = self.base_path / sample_participant / sample_session
        
        key_files = [
            'exp_fatigue.csv',
            'chest_raw_ecg.csv', 
            'forehead_eeg_raw.csv',
            'wrist_hr.csv',
            'exp_nback.csv'
        ]
        
        for filename in key_files:
            file_path = sample_path / filename
            if file_path.exists():
                print(f"\n--- Análisis de {filename} ---")
                try:
                    df = pd.read_csv(file_path)
                    print(f"Dimensiones: {df.shape}")
                    print(f"Columnas: {list(df.columns)}")
                    print(f"Tipos de datos:")
                    print(df.dtypes)
                    print(f"Estadísticas descriptivas (columnas numéricas):")
                    print(df.describe())
                    
                    missing = df.isnull().sum()
                    if missing.sum() > 0:
                        print(f"Valores faltantes:")
                        print(missing[missing > 0])
                    else:
                        print("No hay valores faltantes")
                        
                except Exception as e:
                    print(f"Error al leer {filename}: {e}")
            else:
                print(f"\n--- {filename} no encontrado ---")
    
    def analyze_fatigue_scores(self):
        print("\n=== ANÁLISIS DE SCORES DE FATIGA ===")
        
        fatigue_data = []
        
        for participant in self.participants[:3]:
            for session_level in ['low', 'medium', 'high']:
                session_num = self.session_mapping[participant][session_level]
                file_path = self.base_path / participant / session_num / 'exp_fatigue.csv'
                
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        df['participant'] = participant
                        df['fatigue_level'] = session_level
                        df['session'] = session_num
                        fatigue_data.append(df)
                    except Exception as e:
                        print(f"Error leyendo fatiga para {participant}/{session_num}: {e}")
        
        if fatigue_data:
            combined_fatigue = pd.concat(fatigue_data, ignore_index=True)
            print(f"Datos de fatiga combinados: {combined_fatigue.shape}")
            print("\nColumnas disponibles:")
            print(list(combined_fatigue.columns))
            
            if 'physicalFatigueScore' in combined_fatigue.columns:
                print("\n--- Fatiga Física por Nivel ---")
                physical_stats = combined_fatigue.groupby('fatigue_level')['physicalFatigueScore'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ])
                print(physical_stats)
                
            if 'mentalFatigueScore' in combined_fatigue.columns:
                print("\n--- Fatiga Mental por Nivel ---")
                mental_stats = combined_fatigue.groupby('fatigue_level')['mentalFatigueScore'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ])
                print(mental_stats)
            
            return combined_fatigue
        else:
            print("No se pudieron cargar datos de fatiga")
            return None
    
    def analyze_physiological_signals(self):
        print("\n=== ANÁLISIS DE SEÑALES FISIOLÓGICAS ===")
        
        sample_participant = '01'
        sample_session = '01'
        sample_path = self.base_path / sample_participant / sample_session
        
        ecg_file = sample_path / 'chest_raw_ecg.csv'
        if ecg_file.exists():
            print("\n--- Señal ECG ---")
            df_ecg = pd.read_csv(ecg_file, nrows=1000)
            print(f"Frecuencia de muestreo aproximada: {self.estimate_sampling_rate(df_ecg):.2f} Hz")
            print(f"Duración aproximada (primeras 1000 muestras): {(df_ecg['timestamp'].iloc[-1] - df_ecg['timestamp'].iloc[0])/1000:.2f} segundos")
            print(f"Rango de valores ECG: {df_ecg['ecg_waveform'].min()} - {df_ecg['ecg_waveform'].max()}")
        
        hr_file = sample_path / 'wrist_hr.csv'
        if hr_file.exists():
            print("\n--- Frecuencia Cardíaca ---")
            df_hr = pd.read_csv(hr_file)
            print(f"Número de mediciones HR: {len(df_hr)}")
            print(f"Rango HR: {df_hr['hr'].min():.1f} - {df_hr['hr'].max():.1f} BPM")
            print(f"HR promedio: {df_hr['hr'].mean():.1f} BPM")
        
        eeg_file = sample_path / 'forehead_eeg_raw.csv'
        if eeg_file.exists():
            print("\n--- Señal EEG ---")
            df_eeg = pd.read_csv(eeg_file, nrows=1000)
            print(f"Columnas EEG: {list(df_eeg.columns)}")
            eeg_cols = [col for col in df_eeg.columns if col != 'timestamp']
            if eeg_cols:
                print(f"Número de canales EEG: {len(eeg_cols)}")
                for col in eeg_cols:
                    print(f"  {col}: rango {df_eeg[col].min():.3f} - {df_eeg[col].max():.3f}")
    
    def estimate_sampling_rate(self, df):
        if len(df) < 2:
            return 0
        time_diff = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000
        return (len(df) - 1) / time_diff
    
    def create_summary_report(self):
        print("\n" + "="*60)
        print("REPORTE RESUMEN - ANÁLISIS EXPLORATORIO")
        print("="*60)
        
        print(f"Dataset: FatigueSet")
        print(f"Participantes: {len(self.participants)}")
        print(f"Sesiones por participante: 3 (low, medium, high fatigue)")
        print(f"Total de sesiones: {len(self.participants) * 3}")
        print(f"Tipos de archivos únicos: {len(self.file_types)}")
        
        print(f"\nCategorías de sensores:")
        for category, files in self.categories.items():
            if files:
                print(f"  - {category}: {len(files)} tipos de archivo")
        
        print(f"\nTipos de datos principales:")
        print(f"  - Señales fisiológicas: ECG, EEG, HR, EDA, PPG")
        print(f"  - Datos de movimiento: Acelerómetro, Giroscopio")
        print(f"  - Tareas experimentales: N-back, Task switching, CRT")
        print(f"  - Evaluaciones de fatiga: Física y Mental")
        
        print(f"\nEstructura temporal:")
        print(f"  - Cada sesión contiene múltiples archivos sincronizados")
        print(f"  - Timestamps en formato Unix (milisegundos)")
        print(f"  - Diferentes frecuencias de muestreo por sensor")
    
    def run_complete_analysis(self):
        """Ejecutar análisis exploratorio completo mejorado"""
        print("INICIANDO ANÁLISIS EXPLORATORIO AVANZADO - DATASET FATIGUESET-2")
        print("="*70)
        
        # Análisis básico
        self.load_metadata()
        self.explore_file_structure()
        self.analyze_sample_data()
        fatigue_data = self.analyze_fatigue_scores()
        self.analyze_physiological_signals()
        
        # Análisis avanzado
        self.create_visualizations(fatigue_data)
        self.statistical_analysis(fatigue_data)
        self.analyze_physiological_patterns()
        
        # Reporte final
        self.create_summary_report()
        self.generate_final_report()
        
        return {
            'metadata': self.metadata,
            'file_types': self.file_types,
            'categories': self.categories,
            'fatigue_data': fatigue_data
        }
    
    def create_visualizations(self, fatigue_data):
        """Crear visualizaciones avanzadas"""
        print("\n=== CREANDO VISUALIZACIONES ===")
        
        if fatigue_data is not None and not fatigue_data.empty:
            # Configurar el estilo
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Crear figura con subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Análisis Exploratorio - Dataset de Fatiga', fontsize=16, fontweight='bold')
            
            # 1. Distribución de fatiga física
            if 'physicalFatigueScore' in fatigue_data.columns:
                axes[0,0].hist(fatigue_data['physicalFatigueScore'], bins=20, alpha=0.7, color='red', edgecolor='black')
                axes[0,0].set_title('Distribución Fatiga Física', fontweight='bold')
                axes[0,0].set_xlabel('Puntuación Fatiga Física')
                axes[0,0].set_ylabel('Frecuencia')
                axes[0,0].grid(True, alpha=0.3)
            
            # 2. Distribución de fatiga mental
            if 'mentalFatigueScore' in fatigue_data.columns:
                axes[0,1].hist(fatigue_data['mentalFatigueScore'], bins=20, alpha=0.7, color='blue', edgecolor='black')
                axes[0,1].set_title('Distribución Fatiga Mental', fontweight='bold')
                axes[0,1].set_xlabel('Puntuación Fatiga Mental')
                axes[0,1].set_ylabel('Frecuencia')
                axes[0,1].grid(True, alpha=0.3)
            
            # 3. Correlación fatiga física vs mental
            if 'physicalFatigueScore' in fatigue_data.columns and 'mentalFatigueScore' in fatigue_data.columns:
                axes[0,2].scatter(fatigue_data['physicalFatigueScore'], fatigue_data['mentalFatigueScore'], 
                                alpha=0.6, s=50, c='green', edgecolors='black')
                axes[0,2].set_title('Fatiga Física vs Mental', fontweight='bold')
                axes[0,2].set_xlabel('Fatiga Física')
                axes[0,2].set_ylabel('Fatiga Mental')
                axes[0,2].grid(True, alpha=0.3)
                
                # Añadir línea de tendencia
                z = np.polyfit(fatigue_data['physicalFatigueScore'], fatigue_data['mentalFatigueScore'], 1)
                p = np.poly1d(z)
                axes[0,2].plot(fatigue_data['physicalFatigueScore'], p(fatigue_data['physicalFatigueScore']), 
                             "r--", alpha=0.8, linewidth=2)
                
                # Calcular y mostrar correlación
                corr = fatigue_data['physicalFatigueScore'].corr(fatigue_data['mentalFatigueScore'])
                axes[0,2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0,2].transAxes,
                             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            # 4. Boxplot por nivel de fatiga - Física
            if 'fatigue_level' in fatigue_data.columns and 'physicalFatigueScore' in fatigue_data.columns:
                fatigue_data.boxplot(column='physicalFatigueScore', by='fatigue_level', ax=axes[1,0])
                axes[1,0].set_title('Fatiga Física por Nivel')
                axes[1,0].set_xlabel('Nivel de Fatiga')
                axes[1,0].set_ylabel('Puntuación Fatiga Física')
                axes[1,0].grid(True, alpha=0.3)
            
            # 5. Boxplot por nivel de fatiga - Mental
            if 'fatigue_level' in fatigue_data.columns and 'mentalFatigueScore' in fatigue_data.columns:
                fatigue_data.boxplot(column='mentalFatigueScore', by='fatigue_level', ax=axes[1,1])
                axes[1,1].set_title('Fatiga Mental por Nivel')
                axes[1,1].set_xlabel('Nivel de Fatiga')
                axes[1,1].set_ylabel('Puntuación Fatiga Mental')
                axes[1,1].grid(True, alpha=0.3)
            
            # 6. Evolución temporal de fatiga
            if 'measurementNumber' in fatigue_data.columns and 'physicalFatigueScore' in fatigue_data.columns:
                for participant in fatigue_data['participant'].unique():
                    participant_data = fatigue_data[fatigue_data['participant'] == participant]
                    axes[1,2].plot(participant_data['measurementNumber'], participant_data['physicalFatigueScore'], 
                                 marker='o', alpha=0.7, label=f'P{participant}')
                axes[1,2].set_title('Evolución Fatiga Física', fontweight='bold')
                axes[1,2].set_xlabel('Número de Medición')
                axes[1,2].set_ylabel('Fatiga Física')
                axes[1,2].grid(True, alpha=0.3)
                if len(fatigue_data['participant'].unique()) <= 5:
                    axes[1,2].legend()
            
            plt.tight_layout()
            plt.savefig('fatigue_analysis_complete.png', dpi=300, bbox_inches='tight')
            print("✅ Visualización guardada como 'fatigue_analysis_complete.png'")
            plt.show()
            
        else:
            print("⚠️ No hay datos de fatiga para visualizar")
    
    def statistical_analysis(self, fatigue_data):
        """Realizar análisis estadístico avanzado"""
        print("\n=== ANÁLISIS ESTADÍSTICO AVANZADO ===")
        
        if fatigue_data is not None and not fatigue_data.empty:
            # Test de normalidad
            if 'physicalFatigueScore' in fatigue_data.columns:
                stat, p_value = stats.normaltest(fatigue_data['physicalFatigueScore'].dropna())
                print(f"Test de normalidad - Fatiga Física:")
                print(f"  Estadístico: {stat:.4f}, p-valor: {p_value:.4f}")
                print(f"  {'Normal' if p_value > 0.05 else 'No normal'} (α = 0.05)")
            
            if 'mentalFatigueScore' in fatigue_data.columns:
                stat, p_value = stats.normaltest(fatigue_data['mentalFatigueScore'].dropna())
                print(f"Test de normalidad - Fatiga Mental:")
                print(f"  Estadístico: {stat:.4f}, p-valor: {p_value:.4f}")
                print(f"  {'Normal' if p_value > 0.05 else 'No normal'} (α = 0.05)")
            
            # ANOVA por nivel de fatiga
            if 'fatigue_level' in fatigue_data.columns and 'physicalFatigueScore' in fatigue_data.columns:
                groups = []
                for level in fatigue_data['fatigue_level'].unique():
                    group_data = fatigue_data[fatigue_data['fatigue_level'] == level]['physicalFatigueScore'].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    print(f"\nANOVA - Fatiga Física por nivel:")
                    print(f"  F-estadístico: {f_stat:.4f}, p-valor: {p_value:.4f}")
                    print(f"  {'Diferencias significativas' if p_value < 0.05 else 'No hay diferencias significativas'} entre niveles")
            
            # Correlaciones
            numeric_cols = fatigue_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                print(f"\nMatriz de correlación:")
                corr_matrix = fatigue_data[numeric_cols].corr()
                print(corr_matrix.round(3))
                
                # Visualizar matriz de correlación
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.3f', cbar_kws={"shrink": .8})
                plt.title('Matriz de Correlación - Variables de Fatiga', fontweight='bold')
                plt.tight_layout()
                plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
                print("✅ Matriz de correlación guardada como 'correlation_matrix.png'")
                plt.show()
    
    def analyze_physiological_patterns(self):
        """Analizar patrones en datos fisiológicos"""
        print("\n=== ANÁLISIS DE PATRONES FISIOLÓGICOS ===")
        
        sample_participant = '01'
        sample_session = '01'
        sample_path = self.base_path / sample_participant / sample_session
        
        # Analizar frecuencia cardíaca si está disponible
        hr_file = sample_path / 'wrist_hr.csv'
        if hr_file.exists():
            try:
                df_hr = pd.read_csv(hr_file)
                print(f"Análisis de Frecuencia Cardíaca:")
                print(f"  Número de mediciones: {len(df_hr)}")
                print(f"  HR promedio: {df_hr['hr'].mean():.1f} ± {df_hr['hr'].std():.1f} BPM")
                print(f"  Rango: {df_hr['hr'].min():.1f} - {df_hr['hr'].max():.1f} BPM")
                
                # Crear visualización HR
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(df_hr.index, df_hr['hr'], alpha=0.7, color='red')
                plt.title('Serie Temporal - Frecuencia Cardíaca')
                plt.xlabel('Muestra')
                plt.ylabel('HR (BPM)')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.hist(df_hr['hr'], bins=30, alpha=0.7, color='red', edgecolor='black')
                plt.title('Distribución Frecuencia Cardíaca')
                plt.xlabel('HR (BPM)')
                plt.ylabel('Frecuencia')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('hr_analysis.png', dpi=300, bbox_inches='tight')
                print("✅ Análisis HR guardado como 'hr_analysis.png'")
                plt.show()
                
            except Exception as e:
                print(f"Error analizando HR: {e}")
        
        # Analizar EDA si está disponible
        eda_file = sample_path / 'wrist_eda.csv'
        if eda_file.exists():
            try:
                df_eda = pd.read_csv(eda_file)
                print(f"\nAnálisis de Actividad Electrodérmica (EDA):")
                print(f"  Número de mediciones: {len(df_eda)}")
                print(f"  EDA promedio: {df_eda['eda'].mean():.3f} ± {df_eda['eda'].std():.3f} μS")
                print(f"  Rango: {df_eda['eda'].min():.3f} - {df_eda['eda'].max():.3f} μS")
                
            except Exception as e:
                print(f"Error analizando EDA: {e}")
    
    def generate_final_report(self):
        """Generar reporte final detallado"""
        print("\n" + "="*80)
        print("REPORTE FINAL - ANÁLISIS EXPLORATORIO COMPLETO")
        print("="*80)
        
        report = []
        report.append("DATASET: FatigueSet-2 - Análisis Exploratorio de Datos")
        report.append(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("ESTRUCTURA GENERAL:")
        report.append(f"  • Participantes: {len(self.participants)}")
        report.append(f"  • Sesiones por participante: 3 (low, medium, high)")
        report.append(f"  • Total aproximado de archivos: {len(self.participants) * 3 * len(self.file_types)}")
        report.append(f"  • Tipos de archivo únicos: {len(self.file_types)}")
        report.append("")
        
        report.append("CATEGORÍAS DE SENSORES:")
        for category, files in self.categories.items():
            if files:
                report.append(f"  • {category}: {len(files)} tipos")
                for file in files[:3]:  # Mostrar solo los primeros 3
                    report.append(f"    - {file}")
                if len(files) > 3:
                    report.append(f"    ... y {len(files)-3} más")
        report.append("")
        
        report.append("TIPOS DE DATOS PRINCIPALES:")
        report.append("  • Fisiológicos: ECG, EEG, HR, EDA, PPG, Respiración")
        report.append("  • Movimiento: Acelerómetros, Giroscopios")
        report.append("  • Cognitivos: N-back, Task switching, Tiempo de reacción")
        report.append("  • Subjetivos: Evaluaciones de fatiga física y mental")
        report.append("")
        
        report.append("RECOMENDACIONES PARA ANÁLISIS FUTURO:")
        report.append("  • Sincronizar señales usando timestamps")
        report.append("  • Aplicar filtros de señal para ECG/EEG")
        report.append("  • Analizar correlaciones entre medidas objetivas y subjetivas")
        report.append("  • Implementar análisis de machine learning para predicción")
        report.append("  • Estudiar variabilidad inter-individual")
        
        # Imprimir y guardar reporte
        full_report = "\n".join(report)
        print(full_report)
        
        with open('reporte_eda_completo.txt', 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"\n💾 Reporte completo guardado como 'reporte_eda_completo.txt'")
        print("✅ ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
        print("="*80)

def main():
    """Función principal para ejecutar el análisis completo"""
    print("🚀 ANÁLISIS EXPLORATORIO DATASET FATIGUESET-2")
    print("📊 Analizando datos de fatiga, fisiológicos y cognitivos...")
    print("⏳ Este proceso puede tomar varios minutos...\n")
    
    # Definir la ruta base del dataset
    base_path = "/Users/sebastiannandrealirodriguez/Desktop/fatigueset-2"
    
    # Verificar que existe la ruta
    if not os.path.exists(base_path):
        print(f"❌ ERROR: No se encuentra el directorio {base_path}")
        return None
    
    try:
        # Crear instancia del explorador
        explorer = FatigueDatasetExplorer(base_path)
        
        # Ejecutar análisis completo
        results = explorer.run_complete_analysis()
        
        print("\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print("📈 Archivos generados:")
        print("  • fatigue_analysis_complete.png")
        print("  • correlation_matrix.png") 
        print("  • hr_analysis.png")
        print("  • reporte_eda_completo.txt")
        print("\n💡 Revisa los archivos generados para ver los resultados detallados.")
        
        return results
        
    except Exception as e:
        print(f"❌ ERROR durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()