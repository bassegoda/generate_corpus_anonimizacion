import os
import shutil

# Ruta de la carpeta original
carpeta = r"C:\Users\bassegoda\Documents\GitHub\generate_corpus_anonimizacion\docs_revisar\Octavi"

# Crear una subcarpeta para los archivos ordenados
subcarpeta = os.path.join(carpeta, "archivos_ordenados")
os.makedirs(subcarpeta, exist_ok=True)

# Listar los archivos de la carpeta original
archivos = [f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]

# Ordenar alfabéticamente (estricto)
archivos_ordenados = sorted(archivos)

# Copiar los archivos en orden a la subcarpeta
for i, nombre in enumerate(archivos_ordenados, start=1):
    origen = os.path.join(carpeta, nombre)
    nuevo_nombre = f"{i:03d}_{nombre}"  # Ejemplo: 001_nombre.txt
    destino = os.path.join(subcarpeta, nuevo_nombre)
    
    shutil.copy2(origen, destino)  # copia conservando metadatos (fecha, etc.)

print(f"\n✅ Se copiaron {len(archivos_ordenados)} archivos en orden alfabético a:")
print(subcarpeta)
