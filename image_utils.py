# image_utils.py
import os
from typing import List

def find_image_files(root_dir: str) -> List[str]:
    """
    Busca recursivamente todos los archivos de imagen en un directorio raíz.

    Args:
        root_dir (str): El directorio raíz desde donde comenzar la búsqueda.

    Returns:
        List[str]: Una lista de rutas absolutas a los archivos de imagen encontrados.
    """
    image_paths = []
    # Extensiones de imagen comunes
    supported_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    if not os.path.exists(root_dir):
        print(f"Error: El directorio '{root_dir}' no existe.")
        return []

    print(f"Buscando imágenes en '{root_dir}'...")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    print(f"Encontradas {len(image_paths)} imágenes.")
    return image_paths

if __name__ == '__main__':
    # Ejemplo de uso:
    # Crea un directorio 'test_images' y algunas subcarpetas/imagenes para probar
    # mkdir test_images
    # mkdir test_images/subdir1
    # touch test_images/photo1.jpg
    # touch test_images/subdir1/photo2.png
    
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"Creando directorio de prueba: {test_dir}")
        os.makedirs(os.path.join(test_dir, 'subdir1'), exist_ok=True)
        # Crear archivos ficticios para la prueba
        with open(os.path.join(test_dir, 'example1.jpg'), 'w') as f: f.write('')
        with open(os.path.join(test_dir, 'subdir1', 'example2.png'), 'w') as f: f.write('')

    found_images = find_image_files(test_dir)
    print("\nImágenes encontradas:")
    for img_path in found_images:
        print(img_path)
    print(f"Total: {len(found_images)}")

    # Limpiar directorio de prueba
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
        print(f"\nDirectorio de prueba '{test_dir}' eliminado.")