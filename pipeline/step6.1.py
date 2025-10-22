#!/usr/bin/env python3
"""
PASO 4 DEL PIPELINE: Verificación de anonimización con modelos BSC
Ejecuta los modelos bsc-bio-ehr-es-meddocan y bsc-bio-ehr-es-carmen-anon
sobre los documentos anonimizados para verificar que no detecten entidades.
"""

import os
import json
import csv
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def setup_models():
    """
    Configura y carga ambos modelos BSC.
    
    Returns:
        Tuple: (pipeline_meddocan, pipeline_carmen)
    """
    debug_print("Cargando modelos BSC para verificación...", "INFO")
    
    # Configurar device
    device = 0 if torch.cuda.is_available() else -1
    debug_print(f"Device set to use {'cuda' if torch.cuda.is_available() else 'cpu'}", "INFO")
    
    # Modelo MEDDOCAN
    debug_print("  - Cargando bsc-bio-ehr-es-meddocan...", "DEBUG")
    meddocan_model_path = "models/bsc-bio-ehr-es-meddocan"
    
    try:
        meddocan_tokenizer = AutoTokenizer.from_pretrained(meddocan_model_path)
        
        # Cargar modelo con configuración específica para evitar meta tensors
        meddocan_model = AutoModelForTokenClassification.from_pretrained(
            meddocan_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,  # No usar device_map automático
            trust_remote_code=False
        )
        
        # Verificar que el modelo se cargó correctamente
        if hasattr(meddocan_model, 'config'):
            debug_print(f"      Modelo MEDDOCAN cargado: {meddocan_model.config.model_type}", "DEBUG")
        
        # Crear pipeline sin especificar device en el modelo
        pipeline_meddocan = pipeline(
            "ner",
            model=meddocan_model,
            tokenizer=meddocan_tokenizer,
            aggregation_strategy="simple",
            device=-1  # Forzar CPU para evitar problemas con meta tensors
        )
    except Exception as e:
        debug_print(f"Error cargando modelo MEDDOCAN: {e}", "ERROR")
        raise e
    
    # Modelo CARMEN
    debug_print("  - Cargando bsc-bio-ehr-es-carmen-anon...", "DEBUG")
    carmen_model_path = "models/bsc-bio-ehr-es-carmen-anon"
    
    try:
        carmen_tokenizer = AutoTokenizer.from_pretrained(carmen_model_path)
        
        # Cargar modelo con configuración específica para evitar meta tensors
        carmen_model = AutoModelForTokenClassification.from_pretrained(
            carmen_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,  # No usar device_map automático
            trust_remote_code=False
        )
        
        # Verificar que el modelo se cargó correctamente
        if hasattr(carmen_model, 'config'):
            debug_print(f"      Modelo CARMEN cargado: {carmen_model.config.model_type}", "DEBUG")
        
        # Crear pipeline sin especificar device en el modelo
        pipeline_carmen = pipeline(
            "ner",
            model=carmen_model,
            tokenizer=carmen_tokenizer,
            aggregation_strategy="simple",
            device=-1  # Forzar CPU para evitar problemas con meta tensors
        )
    except Exception as e:
        debug_print(f"Error cargando modelo CARMEN: {e}", "ERROR")
        raise e
    
    debug_print("Modelos cargados exitosamente!", "INFO")
    return pipeline_meddocan, pipeline_carmen

def extract_entities_with_model(text: str, pipeline_model, model_name: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Extrae entidades de un texto usando un modelo específico.
    
    Args:
        text (str): Texto a procesar
        pipeline_model: Pipeline del modelo
        model_name (str): Nombre del modelo para logging
        confidence_threshold (float): Umbral mínimo de confianza
        
    Returns:
        List[Dict]: Lista de entidades encontradas
    """
    debug_print(f"    Analizando con {model_name}...", "DEBUG")
    
    try:
        # Procesar el texto en chunks si es muy largo
        max_length = 512  # Longitud máxima típica para modelos BERT
        chunks = []
        
        if len(text) > max_length:
            # Dividir en chunks con overlap
            words = text.split()
            chunk_size = max_length // 2  # Aproximadamente
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size + 50])  # Overlap de 50 palabras
                chunks.append(chunk)
        else:
            chunks = [text]
        
        all_entities = []
        offset = 0
        
        for i, chunk in enumerate(chunks):
            debug_print(f"      Procesando chunk {i+1}/{len(chunks)}...", "DEBUG")
            
            # Extraer entidades del chunk
            try:
                entities = pipeline_model(chunk)
            except Exception as chunk_error:
                debug_print(f"        ERROR procesando chunk con {model_name}: {chunk_error}", "ERROR")
                continue
            
            # Verificar que entities es una lista válida
            if not isinstance(entities, list):
                debug_print(f"        WARNING: {model_name} devolvió resultado inválido: {type(entities)}", "WARN")
                continue
            
            # Filtrar por confianza y caracteres X, ajustar posiciones
            for entity in entities:
                try:
                    confidence = float(entity.get('score', 0.0))
                    word = entity.get('word', '')
                    start_pos = int(entity.get('start', 0)) + offset
                    end_pos = int(entity.get('end', 0)) + offset
                except (ValueError, TypeError) as parse_error:
                    debug_print(f"        ERROR parseando entidad: {parse_error}", "ERROR")
                    continue
                
                # Extraer texto real para verificar si es solo X
                try:
                    actual_text = text[start_pos:end_pos] if start_pos < len(text) and end_pos <= len(text) else word
                except:
                    actual_text = word
                
                # Filtrar si es solo caracteres X o no cumple umbral de confianza
                if confidence >= confidence_threshold and not is_only_x_characters(actual_text) and not is_only_x_characters(word):
                    entity['start'] = start_pos
                    entity['end'] = end_pos
                    entity['score'] = confidence
                    entity['chunk_id'] = i
                    entity['model'] = model_name
                    all_entities.append(entity)
                else:
                    if is_only_x_characters(actual_text) or is_only_x_characters(word):
                        debug_print(f"        Filtrada detección solo-X: '{actual_text}' / '{word}' (confianza: {confidence:.2f})", "DEBUG")
            
            # Actualizar offset para el siguiente chunk
            offset += len(chunk.split())
        
        debug_print(f"      {model_name}: {len(all_entities)} entidades detectadas (confianza >= {confidence_threshold})", "INFO")
        return all_entities
        
    except Exception as e:
        debug_print(f"      ERROR al procesar con {model_name}: {e}", "ERROR")
        return []

def is_only_x_characters(text: str) -> bool:
    """
    Verifica si un texto contiene solo caracteres 'X' y espacios.
    
    Args:
        text (str): Texto a verificar
        
    Returns:
        bool: True si solo contiene X y espacios
    """
    if not text:
        return False
    
    # Remover espacios y verificar si solo quedan X's
    cleaned_text = text.strip().replace(' ', '')
    return len(cleaned_text) > 0 and all(c == 'X' for c in cleaned_text)

def scan_pii_regex(text: str) -> List[Dict]:
    """
    Escanea el texto en busca de PII residual usando expresiones regulares.
    Busca PII FUERA de marcadores [...] y XXX[ETIQUETA].
    
    Args:
        text (str): Texto a escanear
        
    Returns:
        List[Dict]: Lista de PII detectados con tipo, texto y posición
    """
    # Primero, identificar las zonas protegidas (dentro de marcadores)
    protected_zones = []
    
    # Patrón para [...] 
    for match in re.finditer(r'\[([^\]]+)\]', text):
        protected_zones.append((match.start(), match.end()))
    
    # Patrón para XXX...XXX y XXXXXX
    for match in re.finditer(r'X{3,}', text):
        protected_zones.append((match.start(), match.end()))
    
    def is_in_protected_zone(start: int, end: int) -> bool:
        """Verifica si una posición está dentro de una zona protegida"""
        for pz_start, pz_end in protected_zones:
            if (start >= pz_start and start < pz_end) or (end > pz_start and end <= pz_end):
                return True
        return False
    
    pii_patterns = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'TELEFONO': r'\b(?:\+34|0034)?[6-9]\d{8}\b',
        'DNI': r'\b\d{8}[A-Z]\b',
        'NIE': r'\b[XYZ]\d{7}[A-Z]\b',
        'FECHA_COMPLETA': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'NUMERO_HISTORIA': r'\b(?:NH|HC|HCN)[:\s-]*\d{6,}\b',
        'NUMERO_SEGURIDAD_SOCIAL': r'\b\d{2}[\s-]?\d{10}\b',
    }
    
    detected_pii = []
    
    for pii_type, pattern in pii_patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            
            # Verificar que NO esté en zona protegida
            if not is_in_protected_zone(start, end):
                detected_pii.append({
                    'type': pii_type,
                    'text': match.group(),
                    'start': start,
                    'end': end,
                    'confidence': 1.0  # Regex es determinístico
                })
    
    return detected_pii

def analyze_detected_entities(entities: List[Dict], text: str) -> Dict:
    """
    Analiza las entidades detectadas para clasificarlas.
    Filtra las detecciones que solo contengan caracteres 'X'.
    
    Args:
        entities (List[Dict]): Entidades detectadas
        text (str): Texto original
        
    Returns:
        Dict: Análisis de las entidades
    """
    # Filtrar entidades que solo contengan caracteres X
    filtered_entities = []
    x_only_entities = []
    
    for entity in entities:
        word = entity.get('word', '')
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        
        # Extraer el texto real de la posición
        try:
            actual_text = text[start:end] if start < len(text) and end <= len(text) else word
        except:
            actual_text = word
        
        # Verificar si es solo caracteres X
        if is_only_x_characters(actual_text) or is_only_x_characters(word):
            x_only_entities.append(entity)
            debug_print(f"      Filtrada detección solo-X: '{actual_text}' / '{word}'", "DEBUG")
        else:
            filtered_entities.append(entity)
    
    debug_print(f"      Entidades filtradas (solo-X): {len(x_only_entities)}", "DEBUG")
    debug_print(f"      Entidades válidas restantes: {len(filtered_entities)}", "DEBUG")
    
    # NUEVA: Detección de PII residual con regex
    pii_residual = scan_pii_regex(text)
    
    analysis = {
        'total_entities': len(filtered_entities),
        'filtered_x_entities': len(x_only_entities),
        'entities_by_label': {},
        'entities_by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        'suspicious_entities': [],  # Entidades que no deberían estar ahí
        'xxxxxx_detections': [],   # Detecciones sobre "XXXXXX"
        'other_detections': [],    # Otras detecciones
        'x_only_filtered': x_only_entities,
        'pii_residual': pii_residual,  # NUEVO
        'pii_residual_count': len(pii_residual)  # NUEVO
    }
    
    for entity in filtered_entities:
        label = entity.get('entity_group', entity.get('label', 'UNKNOWN'))
        confidence = entity.get('score', 0.0)
        word = entity.get('word', '')
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        
        # Contar por etiqueta
        if label not in analysis['entities_by_label']:
            analysis['entities_by_label'][label] = 0
        analysis['entities_by_label'][label] += 1
        
        # Contar por confianza
        if confidence >= 0.8:
            analysis['entities_by_confidence']['high'] += 1
        elif confidence >= 0.6:
            analysis['entities_by_confidence']['medium'] += 1
        else:
            analysis['entities_by_confidence']['low'] += 1
        
        # Extraer el texto real de la posición
        try:
            actual_text = text[start:end] if start < len(text) and end <= len(text) else word
        except:
            actual_text = word
        
        entity_info = {
            'label': label,
            'word': word,
            'actual_text': actual_text,
            'confidence': confidence,
            'start': start,
            'end': end,
            'model': entity.get('model', 'unknown')
        }
        
        # Clasificar la detección
        if 'XXXXXX' in actual_text or 'XXXXXX' in word:
            analysis['xxxxxx_detections'].append(entity_info)
        else:
            analysis['other_detections'].append(entity_info)
            # Si no es XXXXXX, es sospechoso (no debería haber entidades reales)
            analysis['suspicious_entities'].append(entity_info)
    
    return analysis

def process_documents_batch(docs_batch: List[str], input_dir: str, output_dir: str, 
                           confidence_threshold: float = 0.7) -> List[Dict]:
    """
    Procesa un lote de documentos con los modelos BSC en paralelo.
    
    Args:
        docs_batch: Lista de IDs de documentos
        input_dir: Directorio de entrada
        output_dir: Directorio de salida
        confidence_threshold: Umbral de confianza
        
    Returns:
        List[Dict]: Resultados del lote
    """
    # Cargar modelos en este worker
    pipeline_meddocan, pipeline_carmen = setup_models()
    
    results = []
    for doc_id in docs_batch:
        result = process_single_document(doc_id, input_dir, pipeline_meddocan, pipeline_carmen, 
                                       output_dir, confidence_threshold)
        results.append(result)
    return results

def process_single_document(doc_id: str, input_dir: str, pipeline_meddocan, pipeline_carmen, 
                           output_dir: str, confidence_threshold: float = 0.7) -> Dict:
    """
    Procesa un documento anonimizado con ambos modelos BSC.
    
    Args:
        doc_id (str): ID del documento
        input_dir (str): Directorio con documentos anonimizados
        pipeline_meddocan: Pipeline del modelo MEDDOCAN
        pipeline_carmen: Pipeline del modelo CARMEN
        output_dir (str): Directorio de salida
        confidence_threshold (float): Umbral de confianza
        
    Returns:
        Dict: Resultado del análisis
    """
    debug_print(f"Verificando anonimización: {doc_id}", "INFO")
    
    # Leer documento anonimizado
    doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_file}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": "Documento no encontrado"
        }
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            anonymized_text = f.read().strip()
    except Exception as e:
        debug_print(f"Error al leer documento {doc_id}: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"Error al leer documento: {e}"
        }
    
    debug_print(f"  Texto anonimizado: {len(anonymized_text)} caracteres", "DEBUG")
    debug_print(f"  Conteo de XXXXXX: {anonymized_text.count('XXXXXX')}", "DEBUG")
    
    # Analizar con MEDDOCAN
    meddocan_entities = extract_entities_with_model(
        anonymized_text, pipeline_meddocan, "MEDDOCAN", confidence_threshold
    )
    
    # Analizar con CARMEN
    carmen_entities = extract_entities_with_model(
        anonymized_text, pipeline_carmen, "CARMEN", confidence_threshold
    )
    
    # Combinar todas las entidades
    all_entities = meddocan_entities + carmen_entities
    
    # Analizar las entidades detectadas
    meddocan_analysis = analyze_detected_entities(meddocan_entities, anonymized_text)
    carmen_analysis = analyze_detected_entities(carmen_entities, anonymized_text)
    combined_analysis = analyze_detected_entities(all_entities, anonymized_text)
    
    # Determinar si la anonimización fue exitosa
    anonymization_success = len(combined_analysis['suspicious_entities']) == 0
    
    # Verificar si hay entidades con confianza muy alta (> 0.99)
    high_confidence_entities = []
    for entity in all_entities:
        confidence = entity.get('score', entity.get('confidence', 0.0))
        if confidence > 0.99:
            high_confidence_entities.append({
                "label": entity.get('entity_group', entity.get('label', 'UNKNOWN')),
                "text": entity.get('word', ''),
                "confidence": confidence,
                "model": entity.get('model', 'UNKNOWN')
            })
    
    should_delete_document = len(high_confidence_entities) > 0
    
    # Crear resultado
    result = {
        "document_id": doc_id,
        "success": True,
        "anonymized_text": anonymized_text,
        "text_length": len(anonymized_text),
        "xxxxxx_count": anonymized_text.count('XXXXXX'),
        "confidence_threshold": confidence_threshold,
        "meddocan_results": {
            "entities_detected": len(meddocan_entities),
            "entities": meddocan_entities,
            "analysis": meddocan_analysis
        },
        "carmen_results": {
            "entities_detected": len(carmen_entities),
            "entities": carmen_entities,
            "analysis": carmen_analysis
        },
        "combined_analysis": combined_analysis,
        "anonymization_verification": {
            "success": anonymization_success,
            "total_detections": len(all_entities),
            "suspicious_detections": len(combined_analysis['suspicious_entities']),
            "xxxxxx_detections": len(combined_analysis['xxxxxx_detections']),
            "other_detections": len(combined_analysis['other_detections'])
        },
        "high_confidence_check": {
            "should_delete": should_delete_document,
            "high_confidence_entities": high_confidence_entities,
            "count": len(high_confidence_entities)
        }
    }
    
    # Guardar resultado detallado
    result_file = os.path.join(output_dir, f"{doc_id}_verification_result.json")
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        debug_print(f"  Resultado guardado: {result_file}", "DEBUG")
    except Exception as e:
        debug_print(f"  ERROR al guardar resultado: {e}", "WARN")
    
    # Mostrar resumen
    debug_print(f"  RESULTADO VERIFICACIÓN:", "INFO")
    debug_print(f"    - Anonimización exitosa: {'SÍ' if anonymization_success else 'NO'}", 
                "INFO" if anonymization_success else "WARN")
    debug_print(f"    - MEDDOCAN detectó: {len(meddocan_entities)} entidades", "INFO")
    debug_print(f"    - CARMEN detectó: {len(carmen_entities)} entidades", "INFO")
    debug_print(f"    - Detecciones sospechosas: {len(combined_analysis['suspicious_entities'])}", 
                "INFO" if len(combined_analysis['suspicious_entities']) == 0 else "WARN")
    
    # Mostrar detecciones sospechosas si las hay
    if combined_analysis['suspicious_entities']:
        debug_print(f"    DETECCIONES SOSPECHOSAS:", "WARN")
        for entity in combined_analysis['suspicious_entities']:
            debug_print(f"      - {entity['label']}: '{entity['actual_text']}' (confianza: {entity['confidence']:.2f})", "WARN")
    
    # Mostrar si hay entidades con confianza muy alta
    if high_confidence_entities:
        debug_print(f"    [!] ALTA CONFIANZA (>0.99): {len(high_confidence_entities)} entidades - DOCUMENTO MARCADO PARA ELIMINACION", "ERROR")
        for entity in high_confidence_entities:
            debug_print(f"      - {entity['label']}: '{entity['text'].strip()}' (confianza: {entity['confidence']:.4f}) [{entity['model']}]", "ERROR")
    
    return result

def create_verification_summary(results: List[Dict], output_dir: str):
    """
    Crea un reporte resumen de la verificación de anonimización.
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        output_dir (str): Directorio de salida
    """
    debug_print("Creando reporte resumen de verificación...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    # Calcular estadísticas globales
    total_meddocan_detections = sum([r.get("meddocan_results", {}).get("entities_detected", 0) for r in successful_results])
    total_carmen_detections = sum([r.get("carmen_results", {}).get("entities_detected", 0) for r in successful_results])
    total_suspicious = sum([r.get("anonymization_verification", {}).get("suspicious_detections", 0) for r in successful_results])
    total_xxxxxx_detections = sum([r.get("anonymization_verification", {}).get("xxxxxx_detections", 0) for r in successful_results])
    
    perfect_anonymizations = len([r for r in successful_results 
                                 if r.get("anonymization_verification", {}).get("success", False)])
    
    # Calcular estadísticas de alta confianza
    high_confidence_docs = [r for r in successful_results 
                           if r.get("high_confidence_check", {}).get("should_delete", False)]
    total_high_confidence_entities = sum([r.get("high_confidence_check", {}).get("count", 0) 
                                         for r in successful_results])
    
    summary = {
        "pipeline_step": 4,
        "process": "anonymization_verification_with_bsc_models",
        "total_documents": len(results),
        "successful_verifications": len(successful_results),
        "failed_verifications": len(failed_results),
        "perfect_anonymizations": perfect_anonymizations,
        "imperfect_anonymizations": len(successful_results) - perfect_anonymizations,
        "global_statistics": {
            "total_meddocan_detections": total_meddocan_detections,
            "total_carmen_detections": total_carmen_detections,
            "total_suspicious_detections": total_suspicious,
            "total_xxxxxx_detections": total_xxxxxx_detections,
            "anonymization_success_rate": (perfect_anonymizations / len(successful_results) * 100) if successful_results else 0
        },
        "high_confidence_statistics": {
            "documents_to_delete": len(high_confidence_docs),
            "total_high_confidence_entities": total_high_confidence_entities,
            "document_ids_to_delete": [r["document_id"] for r in high_confidence_docs]
        },
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step4_verification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 4 - VERIFICACIÓN DE ANONIMIZACIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"Documentos verificados: {summary['total_documents']}")
    print(f"Verificaciones exitosas: {summary['successful_verifications']}")
    print(f"Verificaciones fallidas: {summary['failed_verifications']}")
    print(f"Anonimizaciones perfectas: {summary['perfect_anonymizations']}")
    print(f"Anonimizaciones imperfectas: {summary['imperfect_anonymizations']}")
    print(f"Tasa de éxito de anonimización: {summary['global_statistics']['anonymization_success_rate']:.1f}%")
    print(f"")
    print(f"DETECCIONES DE LOS MODELOS:")
    print(f"  - MEDDOCAN detectó: {summary['global_statistics']['total_meddocan_detections']} entidades")
    print(f"  - CARMEN detectó: {summary['global_statistics']['total_carmen_detections']} entidades")
    print(f"  - Detecciones sospechosas: {summary['global_statistics']['total_suspicious_detections']}")
    print(f"  - Detecciones sobre XXXXXX: {summary['global_statistics']['total_xxxxxx_detections']}")
    
    if summary['imperfect_anonymizations'] > 0:
        print(f"\nDOCUMENTOS CON DETECCIONES SOSPECHOSAS:")
        for result in successful_results:
            if not result.get("anonymization_verification", {}).get("success", False):
                suspicious_count = result.get("anonymization_verification", {}).get("suspicious_detections", 0)
                print(f"  - {result['document_id']}: {suspicious_count} detecciones sospechosas")
    
    # Mostrar documentos marcados para eliminación
    if summary['high_confidence_statistics']['documents_to_delete'] > 0:
        print(f"\n[!] DOCUMENTOS MARCADOS PARA ELIMINACION (Confianza > 0.99):")
        print(f"  Total: {summary['high_confidence_statistics']['documents_to_delete']} documentos")
        print(f"  Entidades de alta confianza detectadas: {summary['high_confidence_statistics']['total_high_confidence_entities']}")
        for doc_id in summary['high_confidence_statistics']['document_ids_to_delete']:
            print(f"    - {doc_id}")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

def create_detailed_detections_report(results: List[Dict], output_dir: str):
    """
    Crea un listado detallado CSV con todas las detecciones sospechosas.
    
    Formato: doc_id, etiqueta, modelo_detector, texto_detectado, confianza
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        output_dir (str): Directorio de salida
    """
    debug_print("Generando listado detallado de detecciones...", "INFO")
    
    detections_list = []
    
    for result in results:
        if not result.get("success", False):
            continue
            
        doc_id = result.get("document_id", "unknown")
        
        # Procesar detecciones de MEDDOCAN
        meddocan_entities = result.get("meddocan_results", {}).get("entities", [])
        for entity in meddocan_entities:
            # Los campos pueden venir como entity_group/score o label/confidence
            etiqueta = entity.get("entity_group", entity.get("label", "UNKNOWN"))
            confianza = entity.get("score", entity.get("confidence", 0.0))
            texto = entity.get("word", "").strip()
            
            detections_list.append({
                "doc_id": doc_id,
                "etiqueta": etiqueta,
                "modelo_detector": "MEDDOCAN",
                "texto_detectado": texto,
                "confianza": f"{confianza:.4f}",
                "posicion_inicio": entity.get("start", ""),
                "posicion_fin": entity.get("end", "")
            })
        
        # Procesar detecciones de CARMEN
        carmen_entities = result.get("carmen_results", {}).get("entities", [])
        for entity in carmen_entities:
            # Los campos pueden venir como entity_group/score o label/confidence
            etiqueta = entity.get("entity_group", entity.get("label", "UNKNOWN"))
            confianza = entity.get("score", entity.get("confidence", 0.0))
            texto = entity.get("word", "").strip()
            
            detections_list.append({
                "doc_id": doc_id,
                "etiqueta": etiqueta,
                "modelo_detector": "CARMEN",
                "texto_detectado": texto,
                "confianza": f"{confianza:.4f}",
                "posicion_inicio": entity.get("start", ""),
                "posicion_fin": entity.get("end", "")
            })
    
    # Ordenar por doc_id y luego por modelo
    detections_list.sort(key=lambda x: (x["doc_id"], x["modelo_detector"]))
    
    # Guardar como CSV
    csv_file = os.path.join(output_dir, "detecciones_detalladas.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            if detections_list:
                fieldnames = ["doc_id", "etiqueta", "modelo_detector", "texto_detectado", 
                            "confianza", "posicion_inicio", "posicion_fin"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detections_list)
        
        debug_print(f"Listado detallado guardado: {csv_file}", "INFO")
        debug_print(f"  Total de detecciones: {len(detections_list)}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar listado detallado: {e}", "ERROR")
    
    # También guardar como JSON para mayor flexibilidad
    json_file = os.path.join(output_dir, "detecciones_detalladas.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_detecciones": len(detections_list),
                "detecciones": detections_list
            }, f, indent=2, ensure_ascii=False)
        debug_print(f"Listado detallado JSON guardado: {json_file}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar listado JSON: {e}", "ERROR")
    
    return detections_list


def process_single_document_light(doc_id: str, input_dir: str, output_dir: str) -> Dict:
    """
    Procesa un documento en modo ligero (regex-only).
    No carga modelos, busca marcadores [**...**], secuencias de X y PII por regex.
    """
    debug_print(f"Verificando (light) anonimización: {doc_id}", "INFO")

    doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_file}", "ERROR")
        return {"document_id": doc_id, "success": False, "error": "Documento no encontrado"}

    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        debug_print(f"Error leyendo documento {doc_id}: {e}", "ERROR")
        return {"document_id": doc_id, "success": False, "error": str(e)}

    # Detectar marcadores [**...**]
    bracketed_pattern = re.compile(r"\[\*\*(.*?)\*\*\]", flags=re.DOTALL)
    bracketed_matches = [m for m in bracketed_pattern.finditer(text)]
    bracketed_count = len(bracketed_matches)

    # Detectar secuencias largas de X
    x_matches = [m for m in re.finditer(r'X{3,}', text)]
    x_count = len(x_matches)

    # Detectar PII residual con regex
    pii_list = scan_pii_regex(text)

    # Construir un resultado compatible con el reporting existente
    combined_analysis = {
        'total_entities': 0,
        'filtered_x_entities': x_count,
        'entities_by_label': {},
        'entities_by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        'suspicious_entities': [],
        'xxxxxx_detections': [{'text': m.group(), 'start': m.start(), 'end': m.end()} for m in x_matches],
        'other_detections': [],
        'x_only_filtered': [],
        'pii_residual': pii_list,
        'pii_residual_count': len(pii_list)
    }

    anonymization_verification = {
        'success': (bracketed_count == 0 and len(pii_list) == 0),
        'total_detections': bracketed_count + len(pii_list) + x_count,
        'suspicious_detections': len(pii_list),
        'xxxxxx_detections': len(x_matches),
        'other_detections': 0
    }

    result = {
        'document_id': doc_id,
        'success': True,
        'anonymized_text': text,
        'text_length': len(text),
        'xxxxxx_count': x_count,
        'confidence_threshold': None,
        'meddocan_results': {'entities_detected': 0, 'entities': [], 'analysis': {}},
        'carmen_results': {'entities_detected': 0, 'entities': [], 'analysis': {}},
        'combined_analysis': combined_analysis,
        'anonymization_verification': anonymization_verification,
        'high_confidence_check': {'should_delete': False, 'high_confidence_entities': [], 'count': 0}
    }

    # Guardar resultado detallado en JSON
    result_file = os.path.join(output_dir, f"{doc_id}_verification_result.json")
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        debug_print(f"  Resultado (light) guardado: {result_file}", "DEBUG")
    except Exception as e:
        debug_print(f"  ERROR al guardar resultado (light): {e}", "WARN")

    # Mostrar un pequeño resumen
    debug_print(f"  (light) bracketed_count={bracketed_count}, pii_residual={len(pii_list)}, x_count={x_count}", "INFO")

    return result

def create_per_document_csv(results: List[Dict], output_dir: str):
    """
    Crea un CSV con métricas detalladas por documento.
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        output_dir (str): Directorio de salida
    """
    debug_print("Generando reporte por documento (CSV)...", "INFO")
    
    csv_file = os.path.join(output_dir, "per_doc.csv")
    
    rows = []
    for result in results:
        # Incluir tanto exitosos como fallidos para capturar errores
        doc_id = result.get("document_id", "unknown")
        verification = result.get("anonymization_verification", {})
        combined = result.get("combined_analysis", {})
        high_conf = result.get("high_confidence_check", {})
        meddocan = result.get("meddocan_results", {}).get("analysis", {})
        carmen = result.get("carmen_results", {}).get("analysis", {})
        error_msg = result.get("error", "")

        row = {
            "doc_id": doc_id,
            "success": result.get("success", False),
            "error": error_msg,
            "anonymization_success": verification.get("success", False),
            "total_detections": verification.get("total_detections", 0),
            "suspicious_detections": verification.get("suspicious_detections", 0),
            "xxxxxx_detections": verification.get("xxxxxx_detections", 0),
            "pii_residual_count": combined.get("pii_residual_count", 0),
            "high_confidence_entities": high_conf.get("count", 0),
            "should_delete": high_conf.get("should_delete", False),
            "meddocan_detections": meddocan.get("total_entities", 0),
            "carmen_detections": carmen.get("total_entities", 0),
            "text_length": result.get("text_length", 0),
            "xxxxxx_count": result.get("xxxxxx_count", 0),
        }
        rows.append(row)
    
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            if rows:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        debug_print(f"Reporte por documento guardado: {csv_file}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar reporte por documento: {e}", "ERROR")

def create_errors_report(results: List[Dict], output_dir: str):
    """
    Genera un CSV/JSON con los documentos que fallaron al procesar y sus mensajes de error.
    """
    debug_print("Generando reporte de errores por documento...", "INFO")
    errors = [r for r in results if not r.get('success', False)]
    csv_file = os.path.join(output_dir, 'errors.csv')
    json_file = os.path.join(output_dir, 'errors.json')

    rows = []
    for e in errors:
        rows.append({
            'doc_id': e.get('document_id', 'unknown'),
            'error': e.get('error', 'unknown')
        })

    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            if rows:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        debug_print(f"Reporte de errores guardado: {csv_file}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar reporte de errores CSV: {e}", "ERROR")

    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({'errors': rows, 'count': len(rows)}, f, indent=2, ensure_ascii=False)
        debug_print(f"Reporte de errores JSON guardado: {json_file}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar reporte de errores JSON: {e}", "ERROR")

def create_markdown_summary(results: List[Dict], output_dir: str, confidence_threshold: float):
    """
    Crea un resumen en formato Markdown legible.
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        output_dir (str): Directorio de salida
        confidence_threshold (float): Umbral de confianza usado
    """
    debug_print("Generando resumen en Markdown...", "INFO")
    
    md_file = os.path.join(output_dir, "summary.md")
    
    successful_results = [r for r in results if r.get("success", False)]
    perfect_anonymizations = len([r for r in successful_results 
                                 if r.get("anonymization_verification", {}).get("success", False)])
    high_confidence_docs = [r for r in successful_results 
                           if r.get("high_confidence_check", {}).get("should_delete", False)]
    
    total_pii_residual = sum([r.get("combined_analysis", {}).get("pii_residual_count", 0) 
                              for r in successful_results])
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("#  Resumen de Validación de Anonimización\n\n")
        f.write(f"**Fecha**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Umbral de confianza**: {confidence_threshold}\n\n")
        
        f.write("##  Estadísticas Generales\n\n")
        f.write(f"- **Total documentos procesados**: {len(results)}\n")
        f.write(f"- **Verificaciones exitosas**: {len(successful_results)}\n")
        percent_perfect = (perfect_anonymizations/len(successful_results)*100) if successful_results else 0.0
        f.write(f"- **Anonimizaciones perfectas**: {perfect_anonymizations} ({percent_perfect:.1f}%)\n")
        f.write(f"- **Documentos con alta confianza (>0.99)**: {len(high_confidence_docs)}\n")
        f.write(f"- **PII residual detectado**: {total_pii_residual} instancias\n\n")
        
        if high_confidence_docs:
            f.write("##  Documentos Marcados para Eliminación\n\n")
            f.write("| Doc ID | Entidades Alta Confianza | Modelos |\n")
            f.write("|--------|--------------------------|--------|\n")
            for doc in high_confidence_docs:
                doc_id = doc["document_id"]
                count = doc.get("high_confidence_check", {}).get("count", 0)
                entities = doc.get("high_confidence_check", {}).get("high_confidence_entities", [])
                models = ", ".join(set([e.get("model", "?") for e in entities]))
                f.write(f"| {doc_id} | {count} | {models} |\n")
            f.write("\n")
        
        if total_pii_residual > 0:
            f.write("## PII Residual Detectado\n\n")
            f.write("Documentos con PII fuera de marcadores:\n\n")
            for result in successful_results:
                pii_list = result.get("combined_analysis", {}).get("pii_residual", [])
                if pii_list:
                    doc_id = result["document_id"]
                    f.write(f"### {doc_id}\n\n")
                    for pii in pii_list:
                        f.write(f"- **{pii['type']}**: `{pii['text']}` (pos: {pii['start']}-{pii['end']})\n")
                    f.write("\n")
        
        f.write("## Archivos Generados\n\n")
        f.write("- `per_doc.csv`: Métricas detalladas por documento\n")
        f.write("- `detecciones_detalladas.csv`: Todas las detecciones de los modelos\n")
        f.write("- `step4_verification_summary.json`: Resumen completo en JSON\n")
    
    debug_print(f"Resumen Markdown guardado: {md_file}", "INFO")

def delete_high_confidence_documents(results: List[Dict], jsonl_file: str, report_only: bool = False):
    """
    Elimina documentos con alta confianza (>0.99) de todas las carpetas del pipeline
    y del archivo JSONL.
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        jsonl_file (str): Ruta al archivo JSONL
        report_only (bool): Si True, solo reporta sin eliminar
    """
    debug_print("Procesando eliminación de documentos con alta confianza...", "INFO")
    
    # Identificar documentos a eliminar
    docs_to_delete = []
    for result in results:
        if not result.get("success", False):
            continue
        
        if result.get("high_confidence_check", {}).get("should_delete", False):
            docs_to_delete.append(result["document_id"])
    
    if not docs_to_delete:
        debug_print("No hay documentos para eliminar", "INFO")
        return {"deleted": 0, "errors": []}
    
    if report_only:
        debug_print(f"[MODO REPORTE] Se eliminarían {len(docs_to_delete)} documentos (no se eliminan realmente)", "WARN")
        return {
            "deleted": 0,
            "documents_to_delete": docs_to_delete,
            "errors": [],
            "report_only": True
        }
    
    debug_print(f"Se eliminarán {len(docs_to_delete)} documentos", "WARN")
    
    deleted_count = 0
    errors = []
    
    # Directorios a limpiar
    directories = [
        "step4_corrected_documents",
        "step4_5_cleaned_documents", 
        "step5_anonymized_documents"
    ]
    
    # Eliminar archivos de cada directorio
    for doc_id in docs_to_delete:
        debug_print(f"  Eliminando documento: {doc_id}", "WARN")
        
        for directory in directories:
            file_path = os.path.join(directory, f"{doc_id}.txt")
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    debug_print(f"    [OK] Eliminado: {file_path}", "DEBUG")
                except Exception as e:
                    error_msg = f"Error eliminando {file_path}: {e}"
                    errors.append(error_msg)
                    debug_print(f"    [X] {error_msg}", "ERROR")
            else:
                debug_print(f"    - No existe: {file_path}", "DEBUG")
    
    # Eliminar entradas del JSONL
    if os.path.exists(jsonl_file):
        debug_print(f"  Actualizando archivo JSONL: {jsonl_file}", "INFO")
        try:
            # Leer todas las líneas
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filtrar líneas que no corresponden a documentos eliminados
            kept_lines = []
            removed_count = 0
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    doc_id = data.get("id", "")
                    
                    if doc_id not in docs_to_delete:
                        kept_lines.append(line)
                    else:
                        removed_count += 1
                        debug_print(f"    [OK] Entrada JSONL eliminada: {doc_id}", "DEBUG")
                except json.JSONDecodeError:
                    # Mantener líneas que no se pueden parsear
                    kept_lines.append(line)
            
            # Guardar archivo actualizado
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                f.writelines(kept_lines)
            
            debug_print(f"   [OK] JSONL actualizado: {removed_count} entradas eliminadas", "INFO")
            
        except Exception as e:
            error_msg = f"Error actualizando JSONL: {e}"
            errors.append(error_msg)
            debug_print(f"    [X] {error_msg}", "ERROR")
    else:
        debug_print(f"  [!] Archivo JSONL no encontrado: {jsonl_file}", "WARN")
    
    # Resumen
    debug_print(f"Eliminación completada:", "INFO")
    debug_print(f"  - Archivos eliminados: {deleted_count}", "INFO")
    debug_print(f"  - Errores: {len(errors)}", "ERROR" if errors else "INFO")
    
    return {
        "deleted": deleted_count,
        "documents_deleted": docs_to_delete,
        "errors": errors
    }

def main():
    parser = argparse.ArgumentParser(description="PASO 4: Verificar anonimización con modelos BSC")
    parser.add_argument("--input-dir", default="step5_anonymized_documents",
                       help="Directorio con documentos anonimizados del paso 3")
    parser.add_argument("--output-dir", default="step6_validation_results",
                       help="Directorio de salida para resultados de verificación")
    parser.add_argument("--confidence-threshold", type=float, default=0.99,
                       help="Umbral mínimo de confianza para considerar detecciones (default: 0.99 = 99%)")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations_fixed.jsonl",
                       help="Archivo JSONL con anotaciones")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar")
    parser.add_argument("--report-only", action="store_true",
                       help="Solo generar reportes sin eliminar archivos (deprecated - use --delete)")
    parser.add_argument("--delete", action="store_true",
                       help="Si está presente, elimina archivos marcados; por defecto NO borra")
    parser.add_argument("--light", action="store_true",
                       help="Modo ligero: NO carga modelos, usa solo regex para detectar PII residual y marcadores. Útil para validar los archivos de step5 rápidamente.")
    parser.add_argument("--strict", action="store_true",
                       help="Modo estricto: exit code diferente si hay documentos dudosos")
    
    args = parser.parse_args()
    
    debug_print("=== PASO 4: VERIFICACIÓN DE ANONIMIZACIÓN CON MODELOS BSC ===", "INFO")
    debug_print(f"Directorio de entrada: {args.input_dir}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    debug_print(f"Umbral de confianza: {args.confidence_threshold}", "INFO")
    
    # Verificar directorio de entrada
    if not os.path.exists(args.input_dir):
        print(f"ERROR: No existe el directorio de entrada: {args.input_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Encontrar documentos a procesar
    input_path = Path(args.input_dir)
    documents = list(input_path.glob("*.txt"))
    
    # Filtrar solo documentos (excluyendo archivos de reporte)
    valid_documents = []
    for doc in documents:
        doc_id = doc.stem
        if not doc_id.endswith('_summary') and not doc_id.endswith('_report'):
            valid_documents.append(doc_id)
    
    if args.max_docs:
        valid_documents = valid_documents[:args.max_docs]
    
    debug_print(f"Encontrados {len(valid_documents)} documentos a verificar", "INFO")
    
    if not valid_documents:
        print("ERROR: No se encontraron documentos válidos para procesar")
        return
    
    # Configurar paralelización (1 worker para evitar problemas con meta tensors)
    num_workers = 1  # Sin paralelización para evitar problemas de carga del modelo
    debug_print(f"Verificación secuencial con {num_workers} worker", "INFO")
    
    # Dividir documentos en lotes
    batch_size = max(1, len(valid_documents) // num_workers)
    doc_batches = [valid_documents[i:i + batch_size] for i in range(0, len(valid_documents), batch_size)]
    
    debug_print(f"Dividido en {len(doc_batches)} lotes de ~{batch_size} documentos cada uno", "INFO")
    
    # Procesar documentos en paralelo
    debug_print("Iniciando verificación paralela...", "INFO")
    results = []
    
    if args.light:
        # Modo ligero: no cargar modelos, procesar secuencialmente con regex
        debug_print("Modo LIGHT activado: procesando sin cargar modelos (regex-only)", "INFO")
        for batch in doc_batches:
            for doc_id in batch:
                try:
                    res = process_single_document_light(doc_id, args.input_dir, args.output_dir)
                    results.append(res)
                except Exception as e:
                    debug_print(f"Error procesando {doc_id} en modo light: {e}", "ERROR")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Enviar lotes a workers
            future_to_batch = {
                executor.submit(process_documents_batch, batch, args.input_dir, args.output_dir, args.confidence_threshold): i 
                for i, batch in enumerate(doc_batches)
            }
            
            # Recopilar resultados
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed_batches += 1
                    
                    successful = len([r for r in batch_results if r.get("success", False)])
                    perfect = len([r for r in batch_results if r.get("success", False) and 
                                  r.get("anonymization_verification", {}).get("success", False)])
                    debug_print(f"  Lote {batch_idx + 1}/{len(doc_batches)} completado: {successful}/{len(batch_results)} exitosos, {perfect} perfectos", "INFO")
                    
                except Exception as e:
                    debug_print(f"Error procesando lote {batch_idx}: {e}", "ERROR")

    
    # Crear reporte resumen
    create_verification_summary(results, args.output_dir)
    
    # Crear listado detallado de detecciones
    create_detailed_detections_report(results, args.output_dir)
    
    # NUEVO: Crear reporte por documento (CSV)
    create_per_document_csv(results, args.output_dir)
    
    # NUEVO: Crear resumen en Markdown
    create_markdown_summary(results, args.output_dir, args.confidence_threshold)
    
    # Eliminar documentos con alta confianza (>0.99)
    # Por defecto no se eliminan; pasar --delete para permitir borrado.
    do_delete = bool(args.delete)
    deletion_result = delete_high_confidence_documents(results, args.jsonl_file, report_only=(not do_delete))

    # Generar listado de errores (documentos que fallaron al procesar)
    create_errors_report(results, args.output_dir)
    
    debug_print("Paso 4 completado! Verificación de anonimización finalizada.", "INFO")
    
    # NUEVO: Exit codes diferenciados en modo strict
    if args.strict:
        # Verificar si hay errores de modelos
        failed_count = len([r for r in results if not r.get("success", False)])
        if failed_count > 0:
            debug_print(f"[STRICT] Errores de modelos detectados: {failed_count}", "ERROR")
            return 1
        
        # Verificar si hay documentos con alta confianza
        high_conf_count = len([r for r in results if r.get("high_confidence_check", {}).get("should_delete", False)])
        if high_conf_count > 0:
            debug_print(f"[STRICT] Documentos con alta confianza detectados: {high_conf_count}", "WARN")
            return 2
    
    return 0

if __name__ == "__main__":
    exit(main())