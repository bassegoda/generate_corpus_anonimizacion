#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para distribuir documentos aleatoriamente desde corpus/anonymized_documents
a las carpetas de revisión, asegurando que cada documento aparezca en exactamente 2 carpetas
y que cada carpeta tenga al menos 120 documentos.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def main():
    # Configuración
    source_dir = Path("corpus/anonymized_documents")
    target_base_dir = Path("docs_revisar")
    min_docs_per_folder = 120
    copies_per_document = 2
    
    # Carpetas de destino
    target_folders = ["David", "Lia", "Elena", "Octavi", "Santiago", "Julian"]
    
    # Verificar que existen los directorios
    if not source_dir.exists():
        print(f"Error: El directorio fuente {source_dir} no existe")
        return
    
    if not target_base_dir.exists():
        print(f"Error: El directorio base {target_base_dir} no existe")
        return
    
    # Obtener todos los documentos .txt del directorio fuente
    source_files = list(source_dir.glob("*.txt"))
    if not source_files:
        print(f"No se encontraron archivos .txt en {source_dir}")
        return
    
    print(f"Encontrados {len(source_files)} documentos en {source_dir}")
    
    # Contador de documentos por carpeta
    docs_count = defaultdict(int)
    
    # Inicializar contadores
    for folder in target_folders:
        folder_path = target_base_dir / folder
        existing_files = list(folder_path.glob("*.txt"))
        docs_count[folder] = len(existing_files)
        print(f"Carpeta {folder}: {docs_count[folder]} documentos existentes")
    
    # Mezclar la lista de archivos para distribución aleatoria
    random.shuffle(source_files)
    
    documents_processed = 0
    
    # Continuar hasta que todas las carpetas tengan al menos min_docs_per_folder documentos
    while any(count < min_docs_per_folder for count in docs_count.values()):
        # Si hemos procesado todos los documentos disponibles, reiniciar la lista
        if documents_processed >= len(source_files):
            print("Reiniciando la lista de documentos para continuar la distribución...")
            random.shuffle(source_files)
            documents_processed = 0
        
        source_file = source_files[documents_processed]
        
        # Encontrar las carpetas que necesitan más documentos
        folders_needing_docs = [folder for folder in target_folders 
                               if docs_count[folder] < min_docs_per_folder]
        
        if len(folders_needing_docs) == 0:
            break
        
        # Seleccionar 2 carpetas aleatorias para este documento
        if len(folders_needing_docs) >= copies_per_document:
            selected_folders = random.sample(folders_needing_docs, copies_per_document)
        else:
            # Si quedan menos carpetas que necesitan documentos, usar todas las disponibles
            # y completar con carpetas aleatorias del resto
            selected_folders = folders_needing_docs.copy()
            remaining_folders = [f for f in target_folders if f not in folders_needing_docs]
            if remaining_folders and len(selected_folders) < copies_per_document:
                additional_needed = copies_per_document - len(selected_folders)
                selected_folders.extend(random.sample(remaining_folders, 
                                                    min(additional_needed, len(remaining_folders))))
        
        # Copiar el documento a las carpetas seleccionadas
        for folder in selected_folders:
            target_path = target_base_dir / folder / source_file.name
            
            # Evitar sobrescribir si ya existe
            counter = 1
            original_target = target_path
            while target_path.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_path = original_target.parent / f"{stem}_{counter}{suffix}"
                counter += 1
            
            try:
                shutil.copy2(source_file, target_path)
                docs_count[folder] += 1
                print(f"Copiado {source_file.name} -> {folder}/ (total: {docs_count[folder]})")
            except Exception as e:
                print(f"Error copiando {source_file.name} a {folder}: {e}")
        
        documents_processed += 1
        
        # Mostrar progreso cada 50 documentos procesados
        if documents_processed % 50 == 0:
            print(f"\n--- Progreso después de {documents_processed} documentos procesados ---")
            for folder in target_folders:
                print(f"{folder}: {docs_count[folder]}/{min_docs_per_folder} documentos")
            print()
    
    # Resumen final
    print("\n" + "="*60)
    print("DISTRIBUCIÓN FINAL:")
    print("="*60)
    total_copies = 0
    for folder in target_folders:
        count = docs_count[folder]
        total_copies += count
        status = "✓ Completo" if count >= min_docs_per_folder else "⚠ Incompleto"
        print(f"{folder:10}: {count:3} documentos {status}")
    
    print(f"\nTotal de copias creadas: {total_copies}")
    print(f"Documentos únicos procesados: {documents_processed}")
    print(f"Promedio de copias por documento: {total_copies/documents_processed:.2f}")

if __name__ == "__main__":
    main()
