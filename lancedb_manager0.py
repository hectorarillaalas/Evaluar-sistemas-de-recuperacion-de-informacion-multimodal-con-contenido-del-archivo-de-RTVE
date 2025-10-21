# lancedb_manager.py
import lancedb
import numpy as np
import pyarrow as pa
import os
from typing import List, Dict, Any, Union
from datetime import datetime, timezone, timedelta

class LanceDBManager:
    def __init__(self, db_path: str = "db"):
        """
        Inicializa el administrador de la base de datos LanceDB.

        Args:
            db_path (str): Ruta al directorio donde se almacenará la base de datos LanceDB.
        """
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        print(f"Conectado a LanceDB en '{self.db_path}'")

    def create_table(self, table_name: str, embedding_dim: int, overwrite: bool = False):
        """
        Crea una nueva tabla en la base de datos LanceDB.

        Args:
            table_name (str): El nombre de la tabla a crear.
            embedding_dim (int): La dimensión de los embeddings a almacenar.
            overwrite (bool): Si es True, sobrescribe la tabla si ya existe.
        """
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("path", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
            pa.field("indexed_at", pa.timestamp('ms'))
        ])

        try:
            if table_name in self.db.table_names():
                if overwrite:
                    print(f"La tabla '{table_name}' ya existe, sobrescribiendo...")
                    self.db.drop_table(table_name)
                    self.table = self.db.create_table(table_name, schema=schema)
                else:
                    print(f"La tabla '{table_name}' ya existe. Usando la tabla existente.")
                    self.table = self.db.open_table(table_name)
            else:
                print(f"Creando nueva tabla '{table_name}' con dimensión de embedding {embedding_dim}.")
                self.table = self.db.create_table(table_name, schema=schema)
            print(f"Tabla '{table_name}' lista.")
            return True
        except Exception as e:
            print(f"Error al crear/abrir la tabla '{table_name}': {e}")
            return False

    def clear_table(self, table_name:str):
        try:
            current_table = self.db.open_table(table_name)
            embedding_dim = None
            for field in current_table.schema.fields:
                if field.name == "vector":
                    if hasattr(field.type, 'list_size'): # For FixedSizeListType
                        embedding_dim = field.type.list_size
                    elif hasattr(field.type, 'num_children') and field.type.num_children > 0: # For ListType
                         # If it's a variable length list, this approach is tricky.
                         # Best to rely on the stored st.session_state.embedding_dim for consistency.
                         pass # Fallback to using the stored session state dim.
            if embedding_dim is None: # Fallback if schema doesn't clearly provide it
                # If LanceDB schema doesn't directly expose embedding_dim on the vector field,
                # you might need to rely on it being stored elsewhere (e.g., in session_state)
                # or infer from existing data before clearing.
                # For simplicity and given your app structure, relying on session_state.embedding_dim is practical.
                embedding_dim = st.session_state.embedding_dim # Use the stored dimension

            self.db.drop_table(table_name)
            print(f"Cleared (dropped) table: {table_name}")
            
            # Recreate the table with the original embedding dimension
            self.create_table(table_name, embedding_dim, overwrite=False) # Overwrite is false here as we just dropped it
            print(f"Recreated empty table: {table_name}")
        except Exception as e:
            st.error(f"Error clearing table '{table_name}': {e}")
            raise # Re-raise the exception to propagate the error if clearing fails

    def get_table(self, table_name: str):
        """
        Obtiene una referencia a una tabla existente.
        """
        if table_name in self.db.table_names():
            return self.db.open_table(table_name)
        else:
            print(f"La tabla '{table_name}' no existe.")
            return None

    def add_images_to_db(self, table_name: str, image_paths: List[str], embeddings: np.ndarray):
        """
        Añade imágenes y sus embeddings a la tabla de la base de datos.

        Args:
            table_name (str): El nombre de la tabla.
            image_paths (List[str]): Lista de rutas de las imágenes.
            embeddings (np.ndarray): Array NumPy de los embeddings correspondientes.
        """
        table = self.get_table(table_name)
        if table is None:
            print(f"No se puede añadir datos: la tabla '{table_name}' no existe o no se pudo abrir.")
            return

        if len(image_paths) != embeddings.shape[0]:
            raise ValueError("El número de rutas de imagen no coincide con el número de embeddings.")

        data = []
        current_max_id = 0
        try:
            # Intentar obtener el ID más alto existente para continuar la secuencia
            # Esto puede fallar si la tabla está vacía, manejar esa excepción
            if table.count_rows() > 0:
                current_max_id = table.to_pandas()["id"].max()
        except Exception:
            current_max_id = -1 # Si falla, es porque está vacía, el primer ID será 0

        for i, (path, embedding) in enumerate(zip(image_paths, embeddings)):
            current_time = datetime.now()
            rounded_microseconds = (current_time.microsecond // 1000) * 1000
            indexed_at_timestamp = current_time.replace(microsecond=rounded_microseconds)
            data.append({
                "id": current_max_id + i + 1,
                "path": path,
                "embedding": embedding.tolist(), # Convertir a lista para PyArrow
                "indexed_at": indexed_at_timestamp
            })
        
        try:
            table.add(data)
            print(f"Añadidos {len(data)} registros a la tabla '{table_name}'.")
        except Exception as e:
            print(f"Error al añadir datos a la tabla '{table_name}': {e}")

    def search_images(self, table_name: str, query_embedding: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Busca imágenes similares en la base de datos usando un embedding de consulta.

        Args:
            table_name (str): El nombre de la tabla a buscar.
            query_embedding (np.ndarray): El embedding de la consulta.
            limit (int): El número máximo de resultados a retornar.

        Returns:
            List[Dict[str, Any]]: Una lista de diccionarios con los resultados (path, distancia, etc.).
        """
        table = self.get_table(table_name)
        if table is None:
            return []
        
        # Convertir el embedding de consulta a un formato compatible si es necesario
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0] # Desenvuelve si es [[...]] a [...]

        try:
            results = table.search(query_embedding).limit(limit).select(["embedding","path"]).to_list()
            print(f"Búsqueda completada. Encontrados {len(results)} resultados en la tabla '{table_name}'.")
            return results
        except Exception as e:
            print(f"Error durante la búsqueda en la tabla '{table_name}': {e}")
            return []

if __name__ == '__main__':
    # Ejemplo de uso:
    db_manager = LanceDBManager(db_path="test_db")
    
    # Necesitas un generador de embeddings para este ejemplo
    from clip_embedding_generator import CLIPEmbeddingGenerator
    clip_gen = CLIPEmbeddingGenerator()
    embedding_dim = clip_gen.get_embedding_dimension()
    
    table_name = "my_image_collection"
    db_manager.create_table(table_name, embedding_dim, overwrite=True)

    # Simular algunas imágenes y embeddings
    sample_paths = ["/path/to/img1.jpg", "/path/to/img2.png"]
    sample_embeddings = np.random.rand(2, embedding_dim).astype(np.float32) # Embeddings aleatorios para la prueba
    sample_embeddings = normalize(sample_embeddings) # Asegurarse de que estén normalizados

    db_manager.add_images_to_db(table_name, sample_paths, sample_embeddings)

    # Simular una búsqueda
    query_text_embedding = clip_gen.get_text_embedding("a beautiful landscape")
    search_results = db_manager.search_images(table_name, query_text_embedding, limit=5)

    print("\nResultados de la búsqueda:")
    for res in search_results:
        print(f"Path: {res['path']}, Distancia: {res['_distance']}")

    # Limpiar base de datos de prueba
    if os.path.exists("test_db"):
        import shutil
        shutil.rmtree("test_db")
        print("\nBase de datos de prueba 'test_db' eliminada.")
