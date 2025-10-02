#!/usr/bin/env python3
"""
PASO 2 DEL PIPELINE: Verificador iterativo de etiquetas faltantes
Verifica si todas las etiquetas del JSON están presentes en el documento.
Si faltan etiquetas, envía a DeepSeek para corrección.
Repite hasta que no haya etiquetas faltantes o se alcance el máximo de iteraciones.
"""

import json
import csv
import argparse
import requests
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import datetime

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_available_labels(csv_file: str) -> Set[str]:
    """Carga las etiquetas disponibles del archivo CSV."""
    labels = set()
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Saltar header
            
            for row in csv_reader:
                if len(row) >= 1 and row[0].strip():
                    labels.add(row[0].strip())
                if len(row) >= 2 and row[1].strip():
                    labels.add(row[1].strip())
    
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {csv_file}")
        return set()
    
    # Remover etiquetas vacías o headers
    labels.discard("")
    labels.discard("MEDDOCAN")
    labels.discard("CARMEN-I")
    
    return labels

def load_all_annotations(jsonl_file: str) -> Dict[str, List[Dict]]:
    """
    Carga todas las anotaciones del archivo JSONL organizadas por ID.
    
    Returns:
        Dict[str, List[Dict]]: Diccionario con ID como clave y lista de entidades como valor
    """
    annotations_by_id = {}
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        doc_id = json_data.get('id', f'unknown_{line_num}')
                        entities = json_data.get('data', [])
                        annotations_by_id[doc_id] = entities
                    except json.JSONDecodeError as e:
                        print(f"Error al parsear JSON en línea {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {jsonl_file}")
        return {}
    
    return annotations_by_id

def clean_entity_names_from_document(document_text: str, available_labels: Set[str]) -> str:
    """
    Elimina nombres literales de entidades del documento.
    
    Args:
        document_text (str): Texto del documento
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        str: Documento limpio sin nombres de entidades
    """
    cleaned_text = document_text
    
    # Lista de nombres de entidades a eliminar
    entity_names_to_remove = [
        "NOMBRE_SUJETO_ASISTENCIA",
        "NOMBRE_PERSONAL_SANITARIO", 
        "ID_SUJETO_ASISTENCIA",
        "CENTRO_SALUD",
        "HOSPITAL",
        "CALLE",
        "TERRITORIO",
        "PAIS",
        "FECHAS",
        "EDAD_SUJETO_ASISTENCIA",
        "SEXO_SUJETO_ASISTENCIA",
        "NUMERO_TELEFONO",
        "CORREO_ELECTRONICO",
        "PROFESION",
        "FAMILIARES_SUJETO_ASISTENCIA",
        "INSTITUCION",
        "URL_WEB",
        "NUMERO_FAX",
        "ID_CONTACTO_ASISTENCIAL",
        "ID_TITULACION_PERSONAL_SANITARIO",
        "ID_EMPLEO_PERSONAL_SANITARIO",
        "ID_ASEGURAMIENTO",
        "NUMERO_BENEF_PLAN_SALUD",
        "IDENTIF_BIOMETRICOS",
        "IDENTIF_DISPOSITIVOS_NRSERIE",
        "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
        "DIREC_PROT_INTERNET",
        "OTROS_SUJETO_ASISTENCIA",
        "OTRO_NUMERO_IDENTIF"
    ]
    
    # Eliminar nombres literales de entidades
    removed_count = 0
    for entity_name in entity_names_to_remove:
        if entity_name in available_labels:
            # Buscar y eliminar el nombre literal de la entidad
            if entity_name in cleaned_text:
                cleaned_text = cleaned_text.replace(entity_name, "")
                removed_count += 1
    
    # Limpiar espacios múltiples y saltos de línea extra
    import re
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    if removed_count > 0:
        debug_print(f"    Eliminados {removed_count} nombres literales de entidades", "DEBUG")
    
    return cleaned_text

def check_missing_entities(document_text: str, expected_entities: List[Dict], available_labels: Set[str]) -> List[Dict]:
    """
    Verifica qué entidades del JSON faltan en el documento.
    
    Args:
        document_text (str): Texto del documento
        expected_entities (List[Dict]): Entidades que deberían estar presentes
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        List[Dict]: Lista de entidades faltantes
    """
    missing_entities = []
    
    for entity in expected_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        # Solo verificar entidades válidas
        if etiqueta in available_labels and texto and texto not in ['[No se proporcionó texto para extraer la edad]', '[valor generado]']:
            # Buscar el texto exacto en el documento (case sensitive)
            if texto not in document_text:
                missing_entities.append(entity)
    
    return missing_entities

def create_correction_prompt(document_text: str, missing_entities: List[Dict], available_labels: Set[str]) -> str:
    """
    Crea un prompt para corregir el documento añadiendo las entidades faltantes.
    
    Args:
        document_text (str): Texto actual del documento
        missing_entities (List[Dict]): Entidades que faltan
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        str: Prompt para DeepSeek
    """
    
    # Extraer textos de entidades faltantes
    missing_texts = []
    missing_labels = set()
    
    for entity in missing_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        if etiqueta in available_labels and texto:
            missing_texts.append(texto)
            missing_labels.add(etiqueta)
    
    missing_labels_list = sorted(list(missing_labels))
    
    prompt = f"""Eres un médico especialista que debe CORREGIR un documento clínico en español. El documento actual está INCOMPLETO porque le faltan algunos datos específicos que DEBEN aparecer.

DOCUMENTO ACTUAL (INCOMPLETO):
{document_text}

DATOS FALTANTES QUE DEBES AÑADIR (usar estos textos EXACTAMENTE como están escritos):
{chr(10).join([f"- {texto}" for texto in missing_texts])}

ETIQUETAS DE LOS DATOS FALTANTES:
{', '.join(missing_labels_list)}

INSTRUCCIONES PARA LA CORRECCIÓN:

1. MANTÉN TODO EL CONTENIDO ACTUAL del documento
2. AÑADE los datos faltantes de manera natural e integrada
3. Los datos faltantes deben aparecer como texto normal, NO como listas o marcadores
4. Integra cada dato faltante en el contexto médico apropiado
5. Si es necesario, crea nuevos párrafos o secciones para incluir los datos
6. Mantén el estilo médico profesional y la coherencia del documento
7. NO elimines ni modifiques el contenido existente
8. NO uses etiquetas XML como <ETIQUETA>texto</ETIQUETA>

RESTRICCIONES CRÍTICAS:
- USA EXACTAMENTE los textos de la lista de datos faltantes
- NO inventes, modifiques o cambies estos textos
- NO añadas información adicional que no esté en la lista
- El documento corregido debe sonar natural y profesional
- Todos los datos faltantes DEBEN aparecer en el documento final

FORMATO DEL TEXTO CORREGIDO:
- Escribe ÚNICAMENTE el documento médico completo y corregido
- NO incluyas explicaciones, comentarios o notas adicionales
- NO uses formato de marcado o etiquetas
- El texto debe fluir naturalmente como un documento médico real

Genera el documento médico COMPLETO Y CORREGIDO que incluya tanto el contenido actual como los datos faltantes."""

    return prompt

def call_deepseek_api(prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """Llama a la API de DeepSeek para generar/corregir el documento."""
    
    url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.3  # Menor temperatura para correcciones más precisas
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        debug_print(f"Error en la llamada a la API: {e}", "ERROR")
        return ""
    except KeyError as e:
        debug_print(f"Error en la respuesta de la API: {e}", "ERROR")
        return ""

def process_document_iteratively(doc_id: str, input_dir: str, expected_entities: List[Dict], 
                                available_labels: Set[str], api_key: str, model: str, 
                                max_iterations: int = 5) -> Dict:
    """
    Procesa un documento iterativamente hasta que todas las entidades estén presentes.
    
    Args:
        doc_id (str): ID del documento
        input_dir (str): Directorio con los documentos del paso 1
        expected_entities (List[Dict]): Entidades que deben estar presentes
        available_labels (Set[str]): Etiquetas disponibles
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        max_iterations (int): Máximo número de iteraciones
        
    Returns:
        Dict: Resultado del procesamiento
    """
    
    debug_print(f"Procesando documento {doc_id}", "INFO")
    
    # Cargar documento inicial
    doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_file}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": "Documento no encontrado",
            "iterations": 0
        }
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            current_document = f.read().strip()
    except Exception as e:
        debug_print(f"Error al leer documento {doc_id}: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"Error al leer documento: {e}",
            "iterations": 0
        }
    
    # Limpiar nombres literales de entidades del documento
    debug_print(f"  Limpiando nombres literales de entidades...", "DEBUG")
    current_document = clean_entity_names_from_document(current_document, available_labels)
    
    # Filtrar entidades válidas esperadas
    valid_expected_entities = []
    for entity in expected_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        if etiqueta in available_labels and texto and texto not in ['[No se proporcionó texto para extraer la edad]', '[valor generado]']:
            valid_expected_entities.append(entity)
    
    debug_print(f"  Entidades esperadas: {len(valid_expected_entities)}", "DEBUG")
    
    iteration_history = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        debug_print(f"  --- ITERACIÓN {iteration} ---", "INFO")
        
        # Verificar entidades faltantes
        missing_entities = check_missing_entities(current_document, valid_expected_entities, available_labels)
        
        debug_print(f"    Entidades faltantes: {len(missing_entities)}", "INFO")
        
        if not missing_entities:
            debug_print(f"    ¡Todas las entidades están presentes! Completado en {iteration-1} iteraciones.", "INFO")
            break
        
        # Mostrar entidades faltantes
        for entity in missing_entities:
            debug_print(f"      Falta: {entity.get('entity', '')} = '{entity.get('text', '')}'", "DEBUG")
        
        # Crear prompt de corrección
        correction_prompt = create_correction_prompt(current_document, missing_entities, available_labels)
        
        # Enviar a DeepSeek para corrección
        debug_print(f"    Enviando a DeepSeek para corrección...", "DEBUG")
        corrected_document = call_deepseek_api(correction_prompt, api_key, model)
        
        if not corrected_document:
            debug_print(f"    Error en la corrección en iteración {iteration}", "ERROR")
            break
        
        # Guardar historial de la iteración
        iteration_info = {
            "iteration": iteration,
            "missing_entities_count": len(missing_entities),
            "missing_entities": missing_entities,
            "document_length_before": len(current_document),
            "document_length_after": len(corrected_document)
        }
        iteration_history.append(iteration_info)
        
        # Actualizar documento actual
        current_document = corrected_document
        debug_print(f"    Documento actualizado (longitud: {len(current_document)} caracteres)", "DEBUG")
    
    # Verificación final
    final_missing = check_missing_entities(current_document, valid_expected_entities, available_labels)
    
    success = len(final_missing) == 0
    debug_print(f"  RESULTADO: {'ÉXITO' if success else 'INCOMPLETO'} - {len(final_missing)} entidades faltantes", 
                "INFO" if success else "WARN")
    
    return {
        "document_id": doc_id,
        "success": success,
        "iterations": iteration,
        "expected_entities_count": len(valid_expected_entities),
        "final_missing_count": len(final_missing),
        "final_missing_entities": final_missing,
        "final_document": current_document,
        "iteration_history": iteration_history,
        "needed_correction": iteration > 1
    }


def clean_failed_documents_from_jsonl(jsonl_file: str, failed_document_ids: List[str]):
    """
    Elimina las referencias de documentos fallidos del archivo JSONL.
    
    Args:
        jsonl_file (str): Ruta al archivo JSONL original
        failed_document_ids (List[str]): Lista de IDs de documentos que fallaron
    """
    if not failed_document_ids:
        debug_print("No hay documentos fallidos para eliminar del JSONL", "INFO")
        return
    
    debug_print(f"Eliminando {len(failed_document_ids)} documentos fallidos del JSONL...", "INFO")
    
    # Crear backup del archivo original
    backup_file = jsonl_file + ".backup"
    import shutil
    shutil.copy2(jsonl_file, backup_file)
    debug_print(f"Backup creado: {backup_file}", "DEBUG")
    
    # Leer todas las líneas y filtrar las que no corresponden a documentos fallidos
    kept_lines = []
    removed_count = 0
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        doc_id = json_data.get('id', f'unknown_{line_num}')
                        
                        if doc_id not in failed_document_ids:
                            kept_lines.append(line)
                        else:
                            removed_count += 1
                            debug_print(f"  Eliminado documento fallido: {doc_id}", "DEBUG")
                            
                    except json.JSONDecodeError as e:
                        debug_print(f"Error al parsear JSON en línea {line_num}: {e}", "WARN")
                        # Mantener líneas con errores de parsing
                        kept_lines.append(line)
        
        # Escribir el archivo limpio
        with open(jsonl_file, 'w', encoding='utf-8') as file:
            for line in kept_lines:
                file.write(line + '\n')
        
        debug_print(f"JSONL limpiado: {removed_count} documentos eliminados, {len(kept_lines)} mantenidos", "INFO")
        
    except Exception as e:
        debug_print(f"Error al limpiar JSONL: {e}", "ERROR")
        # Restaurar backup en caso de error
        shutil.copy2(backup_file, jsonl_file)
        debug_print("Archivo JSONL restaurado desde backup", "WARN")

def create_summary_report(results: List[Dict], output_dir: str):
    """Crea un reporte resumen del paso 2."""
    
    debug_print("Creando reporte resumen del paso 2...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    corrected_docs = [r for r in successful_results if r.get("needed_correction", False)]
    
    summary = {
        "pipeline_step": 2,
        "process": "iterative_verification_and_correction",
        "total_documents": len(results),
        "successful_documents": len(successful_results),
        "failed_documents": len(failed_results),
        "documents_needing_correction": len(corrected_docs),
        "documents_perfect_from_step1": len(successful_results) - len(corrected_docs),
        "average_iterations": sum([r.get("iterations", 0) for r in results]) / len(results) if results else 0,
        "total_entities_expected": sum([r.get("expected_entities_count", 0) for r in results]),
        "total_entities_missing": sum([r.get("final_missing_count", 0) for r in results]),
        "success_rate": len(successful_results) / len(results) * 100 if results else 0,
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step3_verification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 2 - VERIFICACIÓN Y CORRECCIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"Documentos procesados: {summary['total_documents']}")
    print(f"Exitosos: {summary['successful_documents']}")
    print(f"Fallidos: {summary['failed_documents']}")
    print(f"Documentos que necesitaron corrección: {summary['documents_needing_correction']}")
    print(f"Documentos perfectos desde paso 1: {summary['documents_perfect_from_step1']}")
    print(f"Promedio de iteraciones: {summary['average_iterations']:.1f}")
    print(f"Tasa de éxito: {summary['success_rate']:.1f}%")
    print(f"Total entidades esperadas: {summary['total_entities_expected']}")
    print(f"Total entidades aún faltantes: {summary['total_entities_missing']}")
    
    if summary['failed_documents'] > 0:
        print(f"\nDocumentos con problemas:")
        for result in failed_results:
            print(f"  - {result['document_id']}: {result.get('error', 'Error desconocido')}")
        print(f"\nNOTA: Los documentos fallidos han sido eliminados del archivo JSONL original.")
        print(f"Se ha creado un backup del archivo original con extensión .backup")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="PASO 2: Verificar y corregir documentos iterativamente")
    parser.add_argument("--input-dir", default="step3_generated_documents",
                       help="Directorio con documentos del paso 4")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations.jsonl", 
                       help="Archivo JSONL con anotaciones")
    parser.add_argument("--labels-file", default="examples/etiquetas_anonimizacion_meddocan_carmenI.csv",
                       help="Archivo CSV con etiquetas disponibles")
    parser.add_argument("--api-key-file", default="api_keys", 
                       help="Archivo que contiene la clave API de DeepSeek")
    parser.add_argument("--output-dir", default="step4_corrected_documents",
                       help="Directorio de salida para documentos corregidos")
    parser.add_argument("--model", default="deepseek-chat",
                       help="Modelo de DeepSeek a usar")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Máximo número de iteraciones por documento")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar")
    
    args = parser.parse_args()
    
    debug_print("=== PASO 2: VERIFICACIÓN Y CORRECCIÓN ITERATIVA ===", "INFO")
    debug_print(f"Directorio de entrada: {args.input_dir}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    debug_print(f"Máximo iteraciones por documento: {args.max_iterations}", "INFO")
    
    # Verificar directorio de entrada
    if not os.path.exists(args.input_dir):
        print(f"ERROR: No existe el directorio de entrada: {args.input_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar etiquetas disponibles
    debug_print("Cargando etiquetas disponibles...", "INFO")
    available_labels = load_available_labels(args.labels_file)
    debug_print(f"Cargadas {len(available_labels)} etiquetas disponibles", "INFO")
    
    # Cargar anotaciones
    debug_print("Cargando anotaciones del archivo JSONL...", "INFO")
    annotations_by_id = load_all_annotations(args.jsonl_file)
    
    if not annotations_by_id:
        print("ERROR: No se pudieron cargar anotaciones del archivo JSONL")
        return
    
    # Encontrar documentos a procesar
    input_path = Path(args.input_dir)
    documents = list(input_path.glob("*.txt"))
    
    # Filtrar solo documentos que tienen metadatos (excluyendo archivos de reporte)
    valid_documents = []
    for doc in documents:
        doc_id = doc.stem
        if doc_id in annotations_by_id:
            valid_documents.append(doc_id)
    
    if args.max_docs:
        valid_documents = valid_documents[:args.max_docs]
    
    debug_print(f"Encontrados {len(valid_documents)} documentos a verificar", "INFO")
    
    # Cargar clave API
    try:
        with open(args.api_key_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            api_key = None
            for line in lines:
                if line.startswith('deepseek='):
                    api_key = line.split('=', 1)[1].strip()
                    break
            
            if not api_key:
                print("ERROR: No se encontró la clave API de DeepSeek en el archivo")
                return
                
    except FileNotFoundError:
        print(f"ERROR: No se pudo encontrar el archivo de clave API: {args.api_key_file}")
        return
    
    # Procesar cada documento
    results = []
    for i, doc_id in enumerate(valid_documents, 1):
        debug_print(f"\n--- PROCESANDO DOCUMENTO {i}/{len(valid_documents)} ---", "INFO")
        
        expected_entities = annotations_by_id[doc_id]
        
        result = process_document_iteratively(
            doc_id,
            args.input_dir,
            expected_entities,
            available_labels,
            api_key,
            args.model,
            args.max_iterations
        )
        
        results.append(result)
        
        # Guardar documento corregido si fue exitoso
        if result.get("success", False):
            doc_id = result["document_id"]
            doc_file = os.path.join(args.output_dir, f"{doc_id}.txt")
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(result["final_document"])
            debug_print(f"  Documento guardado: {doc_file}", "DEBUG")
        
        # Mostrar progreso cada 3 documentos
        if i % 3 == 0 or i == len(valid_documents):
            successful = len([r for r in results if r.get("success", False)])
            debug_print(f"PROGRESO: {i}/{len(valid_documents)} procesados ({successful} exitosos)", "INFO")
    
    # Crear reporte resumen
    create_summary_report(results, args.output_dir)
    
    # Limpiar documentos fallidos del JSONL
    failed_document_ids = [r["document_id"] for r in results if not r.get("success", False)]
    clean_failed_documents_from_jsonl(args.jsonl_file, failed_document_ids)
    
    debug_print("Paso 2 completado! Pipeline finalizado.", "INFO")

if __name__ == "__main__":
    main()
