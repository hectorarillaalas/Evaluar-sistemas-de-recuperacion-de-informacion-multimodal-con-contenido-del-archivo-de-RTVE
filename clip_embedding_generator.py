# clip_embedding_generator.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Tuple, Union, Generator
import os
from sklearn.preprocessing import normalize 
# from sklearn.preprocessing import normalize # Ya no la importamos aquí

class CLIPEmbeddingGenerator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Inicializa el generador de embeddings CLIP.

        Args:
            model_name (str): Nombre del modelo CLIP a cargar (ej. "openai/clip-vit-base-patch32").
           device (str, optional): Dispositivo a usar ('cuda' o 'cpu'). Si es None, detecta automáticamente.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Cargando CLIP model '{model_name}' en {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        print("Modelo CLIP cargado exitosamente.")

    def generate_embeddings_batch_by_batch(self, image_paths: List[str], batch_size: int = 32) -> Generator[Tuple[int, int, np.ndarray, List[str]], None, None]:
        """
        Calcula los embeddings CLIP para una lista de rutas de imágenes por lotes,
        produciendo el progreso y los embeddings de cada lote.

        Args:
            image_paths (List[str]): Lista de rutas a los archivos de imagen.
            batch_size (int): Tamaño del lote para el procesamiento.

        Yields:
            Tuple[int, int, np.ndarray, List[str]]: Una tupla que contiene:
                - El número de imágenes procesadas hasta el momento.
                - El número total de imágenes a procesar.
                - Un array NumPy de los embeddings del lote actual (NO normalizados aún).
                - Una lista de las rutas de los archivos procesados exitosamente en este lote.
        """
        num_images = len(image_paths)
        processed_count = 0

        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(batch_paths)
            batch_images = []
            current_batch_processed_paths = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(image)
                    current_batch_processed_paths.append(path)
                except Exception as e:
                    # print(f"Advertencia: No se pudo cargar o procesar la imagen {path}: {e}")
                    continue # Saltar esta imagen y continuar con el lote

            if not batch_images:
                # Si el lote entero falló o estaba vacío, simplemente actualizamos el contador y continuamos
                processed_count += len(batch_paths) # Contamos como "procesadas" incluso si fallaron
                yield processed_count, num_images, np.array([]), [] # Yield empty batch if no images were valid
                continue

            try:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    # No normalizamos aquí, la normalización final se hará en la aplicación principal
                    embeddings = self.model.get_image_features(pixel_values=inputs.pixel_values)
                
                processed_count += len(current_batch_processed_paths)
                yield processed_count, num_images, embeddings.cpu().numpy(), current_batch_processed_paths
            except Exception as e:
                print(f"Error en el procesamiento del lote (índice {i}): {e}")
                processed_count += len(batch_paths) # Contamos como "procesadas" incluso si fallaron
                yield processed_count, num_images, np.array([]), [] # Yield empty batch on error

        # El generador terminará cuando se hayan procesado todos los lotes.

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Calcula el embedding CLIP para un texto dado.

        Args:
            text (str): El texto de consulta.

        Returns:
            np.ndarray: El embedding del texto (normalizado).
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        # Normalizamos el embedding de texto aquí porque siempre es uno solo
        return normalize(embedding.cpu().numpy())

    def get_embedding_dimension(self) -> int:
        """
        Retorna la dimensión de los embeddings generados por el modelo CLIP.
        """
        # Generar un embedding de prueba para obtener la dimensión
        dummy_embedding = self.get_text_embedding("dummy text")
        return dummy_embedding.shape[1]

if __name__ == '__main__':
    # Ejemplo de uso: (este bloque solo para probar el módulo, no lo necesita Streamlit)
    test_dir_clip = 'test_images_for_clip'
    if not os.path.exists(test_dir_clip):
        print(f"Creando directorio de prueba para CLIP: {test_dir_clip}. Por favor, añade algunas imágenes aquí.")
        os.makedirs(test_dir_clip, exist_ok=True)
    else:
        print(f"Usando directorio de prueba para CLIP: {test_dir_clip}")

    sample_image_paths = []
    for root, _, files in os.walk(test_dir_clip):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_image_paths.append(os.path.join(root, file))
    
    if not sample_image_paths:
        print("ADVERTENCIA: No se encontraron imágenes en 'test_images_for_clip'. La prueba de embeddings no se ejecutará.")
        print("Por favor, añade algunas imágenes para probar el generador de embeddings.")
    else:
        print(f"Encontradas {len(sample_image_paths)} imágenes de muestra para CLIP.")
        clip_gen = CLIPEmbeddingGenerator()
        
        all_embeddings_collected = []
        all_processed_paths_collected = []

        print("\nIniciando generación de embeddings por lotes...")
        for processed_count, total_count, batch_embeddings, batch_paths in clip_gen.generate_embeddings_batch_by_batch(sample_image_paths, batch_size=2):
            if batch_embeddings.size > 0:
                all_embeddings_collected.append(batch_embeddings)
                all_processed_paths_collected.extend(batch_paths)
            print(f"  Progreso: {processed_count}/{total_count}")

        if all_embeddings_collected:
            final_embeddings = normalize(np.vstack(all_embeddings_collected)) # Normalizar al final
            print(f"\nGeneración de embeddings completada. Total de {len(all_processed_paths_collected)} imágenes procesadas.")
            print(f"Dimensión de los embeddings finales: {final_embeddings.shape[1]}")
            print(f"Primer embedding final (parcial): {final_embeddings[0][:5]}...")
            # print(f"Rutas procesadas: {all_processed_paths_collected}")
        else:
            print("No se generaron embeddings finales.")

        # Prueba de embedding de texto
        text_embedding = clip_gen.get_text_embedding("a dog playing in a park")
        print(f"\nEmbedding de texto ('a dog playing in a park') dimensión: {text_embedding.shape}")
        print(f"Primeros 5 valores del embedding de texto: {text_embedding[0][:5]}...")