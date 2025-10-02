#!/usr/bin/env python3
"""
PASO 1 DEL PIPELINE: Generador directo de documentos médicos
Genera documentos a partir de etiquetas usando el prompt optimizado de DeepSeek
SIN corrección automática - solo una llamada por documento.
"""

import json
import csv
import argparse
import requests
import os
from pathlib import Path
from typing import List, Dict, Set
import datetime

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_available_labels(csv_file: str) -> Set[str]:
    """
    Carga las etiquetas disponibles del archivo CSV.
    
    Args:
        csv_file (str): Ruta al archivo CSV con las etiquetas
        
    Returns:
        Set[str]: Conjunto de etiquetas disponibles
    """
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

def load_all_annotations(jsonl_file: str) -> List[Dict]:
    """
    Carga todos los JSONs del archivo JSONL.
    
    Args:
        jsonl_file (str): Ruta al archivo JSONL
        
    Returns:
        List[Dict]: Lista de todos los JSONs del archivo
    """
    annotations = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        annotations.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error al parsear JSON en línea {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {jsonl_file}")
        return []
    
    return annotations

def create_deepseek_prompt(annotations: List[Dict], available_labels: Set[str]) -> str:
    """
    Crea el prompt optimizado para DeepSeek basado en las anotaciones.
    
    Args:
        annotations (List[Dict]): Lista de anotaciones con etiqueta y texto
        available_labels (Set[str]): Etiquetas disponibles en el diccionario
        
    Returns:
        str: Prompt para DeepSeek
    """
    
    # Extraer solo los textos del JSON y las etiquetas usadas
    entity_texts = []
    used_labels = set()
    
    for annotation in annotations:
        etiqueta = annotation.get('entity', '')
        texto = annotation.get('text', '')
        
        # Verificar que la etiqueta esté en nuestro diccionario y que el texto no esté vacío
        if etiqueta in available_labels and texto.strip():
            entity_texts.append(texto)
            used_labels.add(etiqueta)
    
    # Crear lista de etiquetas que NO debe usar (las que están en el diccionario pero no en este JSON)
    forbidden_labels = available_labels - used_labels
    forbidden_labels_list = sorted(list(forbidden_labels))
    used_labels_list = sorted(list(used_labels))
    
    prompt = f"""Eres un médico especialista redactando un documento clínico en español. Debes crear un texto médico completamente natural y fluido que incorpore de manera orgánica ÚNICAMENTE los siguientes datos específicos:

DATOS EXACTOS A INCLUIR (usar estos textos tal como están escritos):
{chr(10).join([f"- {texto}" for texto in entity_texts if texto.strip()])}

ETIQUETAS PERMITIDAS EN ESTE DOCUMENTO:
{', '.join(used_labels_list)}

ETIQUETAS ESTRICTAMENTE PROHIBIDAS (NO incluir NINGÚN dato de estas categorías):
{', '.join(forbidden_labels_list)}

RESTRICCIÓN CRÍTICA: NO inventes, modifiques o añadas NINGÚN dato personal, identificativo o médico que no esté en la lista exacta de arriba.

INSTRUCCIONES PARA UN DOCUMENTO NATURAL:
1. Redacta como si fueras un médico escribiendo un informe real
2. Usa terminología médica apropiada y estructura profesional
3. Los datos deben aparecer integrados de forma completamente natural en el contexto
4. NO uses listas, viñetas o formatos artificiales
5. El texto debe fluir como prosa médica continua
6. Puedes usar diferentes tipos de documentos: historia clínica, informe de consulta, nota de evolución, etc.
7. Sé creativo con el contexto médico pero SOLO usa los datos proporcionados
8. NO inventes nombres, fechas, diagnósticos, tratamientos o cualquier otra información
9. Si un dato no encaja naturalmente, crea otro párrafo en el texto para incluirlo
10. Añade exploraciones fisicas, resultados de analíticas, resultados de pruebas de imagen, etc. ficticios para darle más realismo al documento

FORMATO DEL TEXTO:
- Escribe ÚNICAMENTE texto médico natural, SIN etiquetas XML o marcadores
- NO uses <ETIQUETA>texto</ETIQUETA> ni ningún formato de marcado
- Los datos deben aparecer como texto normal integrado en el contexto médico
- Ejemplo CORRECTO: "El paciente 12345678A acude al Centro de Salud El Carmen"
- Ejemplo INCORRECTO: "El paciente <ID>12345678A</ID> acude al <CENTRO>Centro de Salud El Carmen</CENTRO>"

RESTRICCIONES CRÍTICAS:
- SOLO los datos de la lista pueden aparecer en el documento
- NO incluyas NINGÚN dato que corresponda a las etiquetas prohibidas
- NO agregues información médica, personal o identificativa adicional
- NO uses etiquetas, marcadores o formato XML/HTML
- El documento debe sonar completamente natural y profesional
- Omite datos vacíos o con texto como "[No se proporcionó...]"

Genera un documento médico natural y profesional que contenga ÚNICAMENTE los datos permitidos."""

    return prompt

def call_deepseek_api(prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """
    Llama a la API de DeepSeek para generar el documento.
    
    Args:
        prompt (str): Prompt para el modelo
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        
    Returns:
        str: Respuesta generada por DeepSeek
    """
    
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
        "max_tokens": 2000,
        "temperature": 0.7
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

def process_single_document(json_data: Dict, available_labels: Set[str], api_key: str, model: str, output_dir: str) -> Dict:
    """
    Procesa un solo JSON y genera su documento médico directamente.
    
    Args:
        json_data (Dict): Datos del JSON
        available_labels (Set[str]): Etiquetas disponibles
        api_key (str): Clave API
        model (str): Modelo a usar
        output_dir (str): Directorio de salida
        
    Returns:
        Dict: Resultado del procesamiento
    """
    
    # Extraer datos del JSON
    if 'data' in json_data:
        annotations = json_data['data']
        json_id = json_data.get('id', 'unknown')
    else:
        annotations = json_data
        json_id = 'unknown'
    
    debug_print(f"Procesando documento {json_id}", "INFO")
    
    # Filtrar solo las etiquetas válidas
    valid_annotations = [ann for ann in annotations if ann.get('entity') in available_labels]
    
    if not valid_annotations:
        debug_print(f"JSON {json_id}: No se encontraron entidades válidas", "WARN")
        return {
            "document_id": json_id,
            "success": False,
            "error": "No se encontraron entidades válidas",
            "entities_count": 0
        }
    
    debug_print(f"  Entidades válidas: {len(valid_annotations)}", "DEBUG")
    
    # Crear prompt optimizado
    prompt = create_deepseek_prompt(valid_annotations, available_labels)
    
    # Generar documento con DeepSeek (una sola llamada)
    debug_print(f"  Generando documento con DeepSeek...", "DEBUG")
    generated_document = call_deepseek_api(prompt, api_key, model)
    
    if not generated_document:
        debug_print(f"JSON {json_id}: Error al generar documento", "ERROR")
        return {
            "document_id": json_id,
            "success": False,
            "error": "Error al generar documento con DeepSeek",
            "entities_count": len(valid_annotations)
        }
    
    # Guardar documento
    output_file = os.path.join(output_dir, f"{json_id}.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(generated_document)
        debug_print(f"  Documento guardado: {output_file}", "INFO")
    except Exception as e:
        debug_print(f"JSON {json_id}: Error al guardar documento: {e}", "ERROR")
        return {
            "document_id": json_id,
            "success": False,
            "error": f"Error al guardar: {e}",
            "entities_count": len(valid_annotations)
        }
    
    # Guardar metadatos del documento
    metadata = {
        "document_id": json_id,
        "generation_step": 1,
        "entities_used": valid_annotations,
        "entities_count": len(valid_annotations),
        "prompt_used": prompt,
        "model_used": model,
        "output_file": output_file
    }
    
    metadata_file = os.path.join(output_dir, f"{json_id}_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        debug_print(f"  Metadatos guardados: {metadata_file}", "DEBUG")
    except Exception as e:
        debug_print(f"Error al guardar metadatos: {e}", "WARN")
    
    return {
        "document_id": json_id,
        "success": True,
        "output_file": output_file,
        "metadata_file": metadata_file,
        "entities_count": len(valid_annotations),
        "document_length": len(generated_document)
    }

def create_summary_report(results: List[Dict], output_dir: str):
    """
    Crea un reporte resumen del procesamiento.
    
    Args:
        results (List[Dict]): Lista de resultados de procesamiento
        output_dir (str): Directorio de salida
    """
    debug_print("Creando reporte resumen...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    summary = {
        "pipeline_step": 1,
        "process": "direct_document_generation",
        "total_documents": len(results),
        "successful_documents": len(successful_results),
        "failed_documents": len(failed_results),
        "total_entities_processed": sum([r.get("entities_count", 0) for r in successful_results]),
        "average_entities_per_doc": sum([r.get("entities_count", 0) for r in successful_results]) / len(successful_results) if successful_results else 0,
        "average_document_length": sum([r.get("document_length", 0) for r in successful_results]) / len(successful_results) if successful_results else 0,
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step1_generation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 1 - GENERACIÓN DIRECTA COMPLETADA")
    print(f"{'='*60}")
    print(f"Documentos procesados: {summary['total_documents']}")
    print(f"Exitosos: {summary['successful_documents']}")
    print(f"Fallidos: {summary['failed_documents']}")
    print(f"Total entidades procesadas: {summary['total_entities_processed']}")
    print(f"Promedio entidades por documento: {summary['average_entities_per_doc']:.1f}")
    print(f"Promedio longitud de documento: {summary['average_document_length']:.0f} caracteres")
    
    if summary['failed_documents'] > 0:
        print(f"\nDocumentos fallidos:")
        for result in failed_results:
            print(f"  - {result['document_id']}: {result.get('error', 'Error desconocido')}")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="PASO 1: Generar documentos médicos directamente desde etiquetas")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations.jsonl", 
                       help="Archivo JSONL con anotaciones")
    parser.add_argument("--labels-file", default="examples/etiquetas_anonimizacion_meddocan_carmenI.csv",
                       help="Archivo CSV con etiquetas disponibles")
    parser.add_argument("--api-key-file", default="api_keys", 
                       help="Archivo que contiene la clave API de DeepSeek")
    parser.add_argument("--output-dir", default="step3_generated_documents",
                       help="Directorio de salida para los documentos generados")
    parser.add_argument("--model", default="deepseek-chat",
                       help="Modelo de DeepSeek a usar")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar (para pruebas)")
    
    args = parser.parse_args()
    
    debug_print("=== PASO 1: GENERACIÓN DIRECTA DE DOCUMENTOS ===", "INFO")
    debug_print(f"Archivo JSONL: {args.jsonl_file}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar etiquetas disponibles
    debug_print("Cargando etiquetas disponibles...", "INFO")
    available_labels = load_available_labels(args.labels_file)
    debug_print(f"Cargadas {len(available_labels)} etiquetas disponibles", "INFO")
    
    # Cargar todos los JSONs
    debug_print("Cargando anotaciones del archivo JSONL...", "INFO")
    all_annotations = load_all_annotations(args.jsonl_file)
    
    if not all_annotations:
        print("ERROR: No se pudieron cargar anotaciones del archivo JSONL")
        return
    
    total_docs = len(all_annotations)
    if args.max_docs:
        total_docs = min(total_docs, args.max_docs)
        all_annotations = all_annotations[:args.max_docs]
    
    debug_print(f"Procesando {total_docs} documentos...", "INFO")
    
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
    for i, json_data in enumerate(all_annotations, 1):
        debug_print(f"\n--- PROCESANDO DOCUMENTO {i}/{total_docs} ---", "INFO")
        
        result = process_single_document(
            json_data, 
            available_labels, 
            api_key, 
            args.model, 
            args.output_dir
        )
        results.append(result)
        
        # Mostrar progreso cada 5 documentos
        if i % 5 == 0 or i == total_docs:
            successful = len([r for r in results if r.get("success", False)])
            debug_print(f"PROGRESO: {i}/{total_docs} procesados ({successful} exitosos)", "INFO")
    
    # Crear reporte resumen
    create_summary_report(results, args.output_dir)
    
    debug_print("Paso 1 completado! Procede con el paso 2 para verificar etiquetas.", "INFO")

if __name__ == "__main__":
    main()
