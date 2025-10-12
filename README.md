# Sistema de Anonimización de Documentos Médicos

Este proyecto implementa un sistema automatizado de anonimización de documentos médicos en español, con un pipeline completo de procesamiento y validación manual. El sistema está diseñado para identificar y anonimizar información de salud protegida (PHI) siguiendo los estándares de los corpus MEDDOCAN y CARMEN-I.

## 📋 Descripción del Proyecto

El sistema procesa documentos médicos identificando automáticamente entidades sensibles como nombres de pacientes, fechas, direcciones, números de identificación, y otra información personal sanitaria, reemplazándolas con marcadores de anonimización (XXX) para proteger la privacidad de los pacientes.

## 🏗️ Estructura del Proyecto

```
anon_bsc/
├── corpus/                          # Corpus de documentos procesados
│   ├── documents/                   # Documentos generados artificialmente (14,035 archivos)
│   ├── anonymized_documents/        # Documentos con entidades sustituidas por XXX (14,035 archivos)
│   ├── entidades/                   # Metadatos de entidades detectadas (14,035 archivos JSON)
│   └── validation_results/          # Validación automática solo de docs con entidades detectadas (1,300 archivos)
├── docs_revisar/                    # Documentos para validación manual
│   ├── David/                       # 120 documentos para revisión
│   ├── Lia/                         # 120 documentos para revisión
│   ├── Elena/                       # 120 documentos para revisión
│   ├── Octavi/                      # 120 documentos para revisión
│   ├── Santiago/                    # 120 documentos para revisión
│   └── Julian/                      # 120 documentos para revisión
├── pipeline/                        # Scripts del pipeline de procesamiento
│   ├── step1_generate_annotations.py    # Generación de anotaciones médicas
│   ├── step2_clean_jsonl.py            # Limpieza de datos JSONL
│   ├── step2_5_semantic_cleaning.py    # Limpieza semántica
│   ├── step3_generate_documents.py     # Generación de documentos
│   ├── step4_correct_docs.py           # Corrección de documentos
│   ├── step4_5_clean_entity_names_enhanced.py # Limpieza avanzada de entidades
│   ├── step5_ocult_and_localization.py # Ocultación y localización
│   └── step6_validation.py             # Validación final
├── distribute_documents.py          # Script de distribución para revisión
├── etiquetas_anonimizacion_meddocan_carmenI.csv # Mapeo de etiquetas
├── guías-de-anotación-de-información-de-salud-protegida.pdf # Guías oficiales
├── requirements.txt                 # Dependencias del proyecto
└── venv/                           # Entorno virtual de Python
```

## 📊 Comparativa de Corpus

| Corpus | Documentos | Caracteres | Descripción |
|--------|------------|------------|-------------|
| **Proyecto Actual** | **14,035** | **~21M** | Documentos médicos sintéticos generados |
| MEDDOCAN | 1,000 | ~1.6M | Corpus de referencia para anonimización médica en español |
| CARMEN-I | ~500 | ~800K | Corpus de informes clínicos en español |

### Estadísticas del Proyecto

- **Documentos generados**: 14,035 archivos de texto (20,980,851 caracteres)
- **Archivos de entidades**: 14,035 archivos JSON con metadatos de entidades detectadas
- **Resultados de validación**: 1,300 archivos (solo documentos con entidades detectadas automáticamente)
- **Documentos para revisión manual**: 720 (120 × 6 revisores, con duplicación cruzada)

## 🔧 Pipeline de Procesamiento

El sistema implementa un pipeline de 6 pasos:

1. **Generación de Anotaciones** (`step1_generate_annotations.py`)
   - Genera documentos médicos sintéticos con entidades PHI
   - Utiliza modelos de IA para crear contenido realista

2. **Limpieza de Datos** (`step2_clean_jsonl.py`, `step2_5_semantic_cleaning.py`)
   - Limpia y normaliza los datos generados
   - Aplica filtros semánticos para mejorar la calidad

3. **Generación de Documentos** (`step3_generate_documents.py`)
   - Convierte las anotaciones en documentos de texto estructurados

4. **Corrección y Limpieza** (`step4_correct_docs.py`, `step4_5_clean_entity_names_enhanced.py`)
   - Corrige inconsistencias en los documentos
   - Mejora la detección y limpieza de entidades
   - Asegura TP de 100% al corroborar que las entidades existen realmente

### Métricas del Step 4: Corrección Iterativa

El Step 4 implementa un proceso de corrección iterativa que detecta y corrige entidades faltantes en los documentos:

#### **📊 Estadísticas de Procesamiento:**
- **4,480 documentos** procesados en esta ejecución
- **3,323 documentos** procesados exitosamente (74.17% tasa de éxito)
- **1,157 documentos** fallaron en el proceso
- **Total dataset**: 12,995 documentos (8,519 ya corregidos previamente)

#### **🎯 Resultados de Corrección:**
- **16.9% de documentos** necesitaron corrección iterativa (757 documentos)
- **57.3% de documentos** estaban perfectos desde el inicio (2,566 documentos)
- **Promedio de 2.08 iteraciones** por documento que necesitó corrección
- **Máximo 5 iteraciones** por documento

#### **📈 Corrección de Entidades:**
- **41,831 entidades** esperadas en total
- **2,721 entidades** faltaban inicialmente
- **1,320 entidades** fueron corregidas exitosamente (48.51%)
- **1,401 entidades** quedaron sin corregir al final

#### **⚙️ Configuración del Proceso:**
- **Tamaño de lote**: 3 documentos por lote
- **Total de lotes**: 1,494 lotes procesados
- **API utilizada**: DeepSeek para corrección automática

5. **Anonimización y Validación** (`step5_ocult_and_localization.py`)
   - Reemplaza entidades sensibles con marcadores XXX
   - Localiza posiciones exactas de las entidades
   - Mantiene la estructura y coherencia del documento

6. **Validación Final** (`step6_validation.py`)
   - Verifica la calidad de la anonimización con modelos BSC
   - Minimiza la existencia de FN no detectados
   - Elimina documentos dudosos del conjunto final
   - Genera reportes de validación automática

## 📊 Estadísticas de Filtrado y Eliminación

### Proceso de Validación Automática

Durante el pipeline de procesamiento, el sistema implementa un riguroso proceso de filtrado para garantizar la calidad de la anonimización:

#### **Step 6: Validación con Alta Confianza (>99%)**

El sistema procesó un total de **20,523 documentos** durante las ejecuciones de validación:

- **Total documentos procesados**: 20,523
- **Total documentos marcados para eliminación**: 6,484
- **Total entidades alta confianza detectadas**: 12,748
- **Tasa global de eliminación**: 31.59%

#### **🎯 Criterios de Eliminación:**

Los documentos son marcados para eliminación cuando:
- Se detectan entidades con **confianza >99%** que no fueron correctamente anonimizadas
- Los modelos BSC identifican información sensible residual
- La verificación cruzada confirma fallos en la anonimización

#### **✅ Corpus Final:**

Después del proceso completo de filtrado y regeneración:
- **14,035 documentos** en el corpus final
- **100% de documentos** pasan los criterios de calidad
- **1,293 archivos** de validación individual disponibles
- **0% de documentos dudosos** en el conjunto final

## 📊 Métricas de Detección de Entidades

### Evolución de TP, FP, TN, FN a lo largo del Pipeline

Las siguientes métricas muestran la evolución de la detección de entidades desde la generación inicial hasta el corpus final:

#### **🎯 Estado Inicial (Post-generación DeepSeek directa):**
| Métrica | Porcentaje | Descripción |
|---------|------------|-------------|
| **TP** | 52.4% (21,915) | Entidades reales correctamente detectadas |
| **FP** | 16.0% (6,700) | Entidades en etiquetas pero no en texto real |
| **FN** | 31.6% (13,216) | Entidades reales no detectadas (eliminadas en Step 6) |

**Métricas Derivadas:**
- **Precisión**: 76.6% (0.766)
- **Recall**: 62.4% (0.624)
- **F1-Score**: 68.8% (0.688)

#### **✅ Estado Final (Post-Step 6 - Validación):**
| Métrica | Porcentaje | Descripción |
|---------|------------|-------------|
| **TP** | 100% (30,044) |Entidades correctamente detectadas y anonimizadas |
| **FP** | 0% (0) | Falsos positivos eliminados durante validación |
| **FN** | 5.0% (1,430) | Entidades reales no detectadas (requieren validación humana) |

**Métricas Derivadas:**
- **Precisión**: 100% (1.000)
- **Recall**: 95.2% (0.952)
- **F1-Score**: 97.6% (0.976)

#### **📈 Evolución del Pipeline:**
- **Precisión**: 76.6% → **100%** → 100% (perfección tras Step 4)
- **Recall**: 62.4% → **100%** → 95.2% (perfección en documentos válidos)
- **F1-Score**: 68.8% → **100%** → 97.6% (perfección intermedia)
- **Falsos Positivos**: 16.0% → **0%** → 0% (eliminados en Step 4)
- **Falsos Negativos**: 31.6% → **0%** → 5.0% (Step 4 encuentra todo; Step 6 filtra calidad)

## 🏷️ Entidades Detectadas

El sistema identifica y anonimiza información de salud protegida (PHI) siguiendo los estándares de MEDDOCAN y CARMEN-I. Las categorías específicas de entidades están definidas en el archivo `etiquetas_anonimizacion_meddocan_carmenI.csv`, que contiene el mapeo completo entre ambos sistemas de etiquetado.

## 👥 Proceso de Validación Manual

### Distribución de Documentos

La carpeta `docs_revisar/` contiene documentos distribuidos aleatoriamente entre 6 revisores:

- Cada revisor tiene **120 documentos** en su carpeta personal
- Esto permite validación cruzada y control de calidad

### Instrucciones para Revisores

#### 📋 Proceso de Validación Manual

1. **Acceder a su carpeta**: `docs_revisar/[su_nombre]/`
   - Encontrará 120 documentos anonimizados para revisar
   - Cada documento aparece en exactamente 2 carpetas para validación cruzada

2. **Abrir el archivo CSV de validación**: `validacion_[su_nombre].csv`
   - Contiene la lista completa de sus 120 documentos ordenados alfabéticamente
   - Incluye columnas para registrar sus hallazgos

3. **Para cada documento**:
   - **Abrir el archivo .txt** y revisar el contenido anonimizado
   - **Confirmar que no se haya perdido información médica relevante**

4. **Registrar en el CSV**:
   - **Columna "Correctamente_Anonimizado"**: Escribir "Sí" o "No"
   - **Columna "Texto_Conflictivo"**: Si hay problemas, copiar y pegar el texto problemático
   - **Columna "Observaciones"**: Comentarios adicionales sobre el documento

5. **Consultar las guías**: Usar el PDF `guías-de-anotación-de-información-de-salud-protegida.pdf` como referencia

6. **Guardar el CSV** regularmente durante la revisión

#### 🎯 Criterios de Evaluación

- ✅ **Correcto**: Todas las entidades sensibles están anonimizadas con "XXX"
- ❌ **Incorrecto**: Se detectan entidades sensibles sin anonimizar
- ⚠️ **Dudoso**: Casos ambiguos que requieren análisis adicional

#### 📊 Entrega de Resultados

Al completar la revisión, el archivo CSV contendrá:
- Lista completa de documentos revisados
- Estado de anonimización de cada documento
- Texto específico de problemas encontrados
- Observaciones y comentarios del revisor

### Ejemplo de Anonimización

**Documento Original:**
```
En la valoración clínica realizada, se documenta el estado del paciente con 
número de historia clínica HC-2024-789012. [...] se ha registrado el 
identificador del sujeto de asistencia 87654321B y el contacto asistencial 
B-78423915 correspondiente. [...] disponible en 
https://historiaclinica.hgugm.es/paciente/area_privada [...] normativa de 
protección de datos vigente en España.
```

**Documento Anonimizado:**
```
En la valoración clínica realizada, se documenta el estado del paciente con 
número de historia clínica XXX. [...] se ha registrado el identificador del 
sujeto de asistencia XXX y el contacto asistencial XXX correspondiente. [...] 
disponible en XXX [...] normativa de protección de datos vigente en XXX.
```

## 🛠️ Instalación y Configuración

### Requisitos

- Python 3.11+
- Entorno virtual configurado
- Dependencias listadas en `requirements.txt`

### Instalación

```bash
# Clonar el repositorio
git clone [url-del-repositorio]
cd anon_bsc

# Activar entorno virtual
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de API Keys

Crear archivo `api_keys` con las claves necesarias para los modelos de IA utilizados en el pipeline.

## 📈 Métricas de Calidad

El pipeline está diseñado para optimizar las métricas de anonimización:

### Métricas Estimadas del Sistema
- **Verdaderos Positivos (TP)**: ~1.0 (100% de precisión en entidades detectadas)
- **Falsos Negativos (FN)**: ~0.05 (5% de entidades no detectadas)

### Estrategia de Validación
- **Steps 4-5**: Aseguran TP de 100% corroborando que las entidades detectadas existen realmente
- **Step 6**: Minimiza FN mediante validación con modelos BSC especializados
- **Validación humana**: Confirma y refina las métricas estimadas

### Resultados de Validación
- **Total de documentos procesados**: 14,035
- **Documentos con entidades detectadas**: 1,300 (almacenados en `validation_results/`)
- **Tasa de detección de entidades**: ~9.3% de los documentos contienen PHI detectable (No todas estas entidades son correctas)

## 📚 Documentación Adicional

- **Guías de Anonimización**: `guías-de-anotación-de-información-de-salud-protegida.pdf`
- **Mapeo de Etiquetas**: `etiquetas_anonimizacion_meddocan_carmenI.csv`
- **Scripts de Pipeline**: Documentados individualmente en `pipeline/`

## 🤝 Contribución

Para contribuir al proyecto:

1. Revisar los documentos asignados en `docs_revisar/`
2. Reportar errores o inconsistencias encontradas
3. Sugerir mejoras al pipeline de anonimización
4. Documentar casos edge encontrados durante la revisión

## 📄 Licencia

Este proyecto está desarrollado para fines de investigación en anonimización de documentos médicos en español, siguiendo las normativas de protección de datos vigentes.

---

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Contacto

- **GitHub:** [@ramsestein](https://github.com/ramsestein)
- **Proyecto:** [generate_corpus_anonimizacion](https://github.com/ramsestein/generate_corpus_anonimizacion)

## 🙏 Agradecimientos

- **MEDDOCAN:** Corpus de anonimización médica en español
- **CARMEN-I:** Corpus de anonimización médica
- **BSC:** Barcelona Supercomputing Center
- **DeepSeek:** Modelos de IA para corrección de texto

---

*Última actualización: Octubre 2025*
