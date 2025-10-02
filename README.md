# Generador de Corpus de Anonimización Médica

Este proyecto implementa un pipeline completo para generar, corregir y anonimizar documentos médicos en español, creando un corpus de entrenamiento para sistemas de anonimización.

## 🏥 Descripción

El sistema genera documentos médicos sintéticos basados en anotaciones reales del corpus MEDDOCAN y CARMEN-I, los corrige iterativamente usando IA, y los anonimiza preservando las localizaciones de las entidades sensibles.

## 📋 Pipeline de Procesamiento

### Paso 1: Generación de Documentos (`step1_generate_annotations.py`)
- Genera anotaciones médicas sintéticas basadas en patrones reales
- Utiliza modelos de IA para crear documentos médicos coherentes
- Produce archivos JSONL con entidades etiquetadas

### Paso 2: Limpieza Semántica (`step2_5_semantic_cleaning.py`)
- Limpia y valida las anotaciones generadas
- Elimina inconsistencias y duplicados
- Mejora la calidad semántica del corpus

### Paso 3: Generación de Documentos (`step3_generate_documents.py`)
- Convierte anotaciones en documentos médicos completos
- Genera texto médico profesional y coherente
- Crea documentos de longitud apropiada (~220 palabras promedio)

### Paso 4: Corrección Iterativa (`step4_correct_docs.py`)
- Verifica que todas las entidades estén presentes en el documento
- Corrige documentos faltantes usando DeepSeek
- Elimina referencias de documentos fallidos del JSONL
- Limpia etiquetas literales de entidades

### Paso 5: Anonimización (`step5_ocult_and_localization.py`)
- Reemplaza entidades sensibles por "XXX"
- Calcula posiciones exactas de entidades (original y anonimizada)
- Actualiza el JSONL con localizaciones para entrenamiento

### Paso 6: Validación (`step6_validation.py`)
- Valida la calidad del corpus generado
- Verifica cobertura de entidades
- Genera métricas de evaluación

## 🚀 Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/ramsestein/generate_corpus_anonimizacion.git
cd generate_corpus_anonimizacion
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar claves API:**
```bash
# Crear archivo api_keys con tus claves
echo "deepseek=tu_clave_deepseek" > api_keys
```

## 🔧 Uso

### Ejecutar pipeline completo:
```bash
python step1_generate_annotations.py
python step2_clean_jsonl.py
python step2_5_semantic_cleaning.py
python step3_generate_documents.py
python step4_correct_docs.py
python step5_ocult_and_localization.py
python step6_validation.py
```

### Ejecutar paso específico:
```bash
python step4_correct_docs.py --max-docs 5 --max-iterations 3
```

## 📁 Estructura del Proyecto

```
├── step1_generate_annotations.py      # Generación de anotaciones
├── step2_5_semantic_cleaning.py       # Limpieza semántica
├── step2_clean_jsonl.py              # Limpieza de JSONL
├── step3_generate_documents.py       # Generación de documentos
├── step4_correct_docs.py             # Corrección iterativa
├── step5_ocult_and_localization.py    # Anonimización
├── step6_validation.py                # Validación final
├── examples/                          # Datos de ejemplo
│   ├── etiquetas_anonimizacion_meddocan_carmenI.csv
│   └── jsonl_data/
│       └── medical_annotations.jsonl
├── models/                            # Modelos preentrenados
│   ├── bsc-bio-ehr-es-carmen-anon/
│   ├── bsc-bio-ehr-es-meddocan/
│   └── labse/
└── README.md
```

## 🤖 Modelos Utilizados

- **DeepSeek Chat:** Corrección iterativa de documentos
- **BSC Bio EHR ES:** Modelos especializados en texto médico español
- **LaBSE:** Embeddings multilingües para similitud semántica

## 🔒 Privacidad y Seguridad

- Todos los documentos son sintéticos
- No se utilizan datos médicos reales
- Las entidades se anonimizan completamente
- El corpus es seguro para investigación

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Contacto

- **Autor:** Ramses Stein
- **GitHub:** [@ramsestein](https://github.com/ramsestein)
- **Proyecto:** [generate_corpus_anonimizacion](https://github.com/ramsestein/generate_corpus_anonimizacion)

## 🙏 Agradecimientos

- **MEDDOCAN:** Corpus de anonimización médica en español
- **CARMEN-I:** Corpus de anonimización médica
- **BSC:** Barcelona Supercomputing Center
- **DeepSeek:** Modelos de IA para corrección de texto
