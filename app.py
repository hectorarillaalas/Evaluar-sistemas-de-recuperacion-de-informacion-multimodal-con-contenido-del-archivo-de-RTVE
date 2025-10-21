# app.py
import streamlit as st
import lancedb
import os
import shutil
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from sklearn.preprocessing import normalize # AÑADIDO: Importar normalize aquí

# Para el diálogo de selección de carpeta (solo en ejecución local)
from tkinter import Tk, filedialog

# Importar los módulos que creamos
from image_utils import find_image_files
from clip_embedding_generator import CLIPEmbeddingGenerator

from lancedb_manager import LanceDBManager

from multilingual_clip import pt_multilingual_clip
import transformers

## Extraer embeddings de M-CLIP
import torch
#import open_clip #(#)
import requests
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32") #(#)
#model.to(device) #(#)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
#image = preprocess(image).unsqueeze(0).to(device) #(#)

#with torch.no_grad(): #(#)
#    image_features = model.encode_image(image) #(#)

#print("Image features shape:", image_features.shape) #(#)

####### INICIO DE SELECCION DE MODELO CLIP ###############
# Configuración inicial de modelos CLIP
MODEL_CONFIGS = {
    "CLIP": {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "processor": "clip"
    },
    "M-CLIP": {
        "model_name": "M-CLIP/XLM-Roberta-Large-Vit-L-14",
        "pretrained": "multilingual-clip",
        "processor": "multilingual-clip"
    }
}

# Carga de modelos con cache
@st.cache_resource
def load_clip_model(model_type="CLIP"):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Modelo {model_type} no está configurado. Opciones válidas: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import open_clip
#        from multilingual_clip import pt_multilingual_clip
        from transformers import AutoTokenizer

        if model_type == "CLIP":
            model, _, preprocess = open_clip.create_model_and_transforms(
                config["model_name"],
                pretrained=config["pretrained"]
            )
            processors = {
                'text' == None,
                'image' == preprocess
            }
        else:  # M-CLIP
#            from multilingual_clip import pt_multilingual_clip
#            from transformers import AutoTokenizer
            try:
                model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
                    config["model_name"],
                    use_auth_token=True  # Usa el token guardado
                )
            except Exception as e:
                # Opción 2: Descarga manual alternativa
                st.warning("Descargando modelo M-CLIP desde repositorio alternativo...")
                model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
                    MODEL_CONFIGS["M-CLIP"]["model_name"]
                )
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

            _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
            
            # Empaquetamos preprocesadores
            processors = {
                'text': tokenizer,
                'image': preprocess
            }
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device), processors
    except Exception as e:
        st.error(f"Error cargando {model_type}:{str(e)}")
        return None, None
####### FIN DE SELECCION DE MODELO CLIP ###############


# --- Funciones Auxiliares para Tkinter (solo para ejecución local) ---
def choose_directory_dialog() -> str:
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def choose_file_dialog():
    """
    Abre un diálogo para seleccionar un archivo (no un directorio)
    """
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    root.attributes('-topmost', True)  # Traer al frente
    
    file_path = filedialog.askopenfilename(
        title="Selecciona el documento con URLs",
        filetypes=[
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(
    page_title="Organizador de Fotos con CLIP & LanceDB",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Variables Globales / de Sesión ---
DB_PATH = "db"
LANCEDB_TABLE_NAME = "image_embeddings"
BATCH_SIZE_INDEXING = 32 # Tamaño de lote para la indexación
SEARCH_LIMIT_DEFAULT = 10 # Límite por defecto para resultados de búsqueda
LANCEDB_MODEL_TYPE = "CLIP" # Modelo de indexación


# Inicializar generador CLIP y LanceDB Manager en el estado de sesión
if 'clip_generator' not in st.session_state:
    st.session_state.clip_generator = CLIPEmbeddingGenerator()
    st.session_state.embedding_dim = st.session_state.clip_generator.get_embedding_dimension()
    st.session_state.lancedb_manager = LanceDBManager(db_path=DB_PATH)
#    st.session_state.lancedb_manager.create_table(LANCEDB_TABLE_NAME, st.session_state.embedding_dim, overwrite=False)

#Inicializar lancedb
if 'lancedb_manager' not in st.session_state:
    st.session_state.lancedb_manager = LanceDBManager(db_path=DB_PATH)

# Inicializar la ruta del directorio de indexación en session_state
if 'indexing_root_dir_value' not in st.session_state:
    st.session_state.indexing_root_dir_value = os.getcwd()

# Inicializar la ruta del documento de indexación en session_state
if 'indexing_root_urls_value' not in st.session_state:
    st.session_state.indexing_root_urls_value = os.getcwd()

# Inicializar la tabla seleccionada
if 'tabla_seleccionada' not in st.session_state:
    st.session_state.tabla_seleccionada=None

# --- Funciones Auxiliares para UI ---
def display_image_results(results: list):
    if not results:
        st.info("No se encontraron resultados para tu búsqueda.")
        return

    st.subheader(f"Resultados Encontrados ({len(results)}):")
    cols = st.columns(5) # Mostrar 5 imágenes por fila

    for i, res in enumerate(results):
        with cols[i % 5]:
            try:
                img_path = res['path']
                image = Image.open(img_path)
                st.image(image, caption=f"Distancia: {res['_distance']:.4f}", use_container_width=True)
                st.markdown(f"<small>{os.path.basename(img_path)}</small>", unsafe_allow_html=True)
            except FileNotFoundError:
                st.warning(f"Imagen no encontrada en disco: {res['path']}")
            except Exception as e:
                st.error(f"Error al cargar la imagen {res['path']}: {e}")

# --- Interfaz de Usuario ---
st.title("📸 Organizador de Fotos Semántico")
st.markdown("Busca tus fotos usando texto o imágenes, impulsado por **CLIP** y **LanceDB**.")
tab0,tab1,tab2,tab3=st.tabs(["💻 Crear BD","⚙️ Gestión BD","🔍 Busqueda", "📐 Operaciones"])

if 'vector_nombres_tabla' not in st.session_state:
    try:
        # Conectar a la base de datos LanceDB
        db = lancedb.connect(DB_PATH)
        
        # Obtener nombres de tablas disponibles
        nombres_tablas = db.table_names()
        
        # Procesar cada nombre de tabla
        tablas_procesadas = []
        for nombre_tabla in nombres_tablas:
            if '_' in nombre_tabla:
                # Dividir en el último guión bajo
                partes = nombre_tabla.rsplit('_', 1)
                tablas_procesadas.append((partes[0], partes[1]))
            else:
                # Si no tiene guión, usar nombre completo y modelo por defecto
                tablas_procesadas.append((nombre_tabla, "CLIP"))
        
        # Guardar en session state
        st.session_state.vector_nombres_tabla = tablas_procesadas
        
    except Exception as e:
        st.error(f"Error inicializando tablas: {str(e)}")
        st.session_state.vector_nombres_tabla = []

#    st.session_state.vector_nombres_tabla = (lancedb.connect(DB_PATH).table_names.rsplit('_',1)[0](),lancedb.connect(DB_PATH).table_names.rsplit('_',1)[1]())

# Tag para crear nuevas imágenes
with tab0:
    st.header("Opcion para crear tablas nuevas")
    
    nombre_tabla = st.text_input("Escriba el nombre de la tabla nueva")

    #Mostrar opciones de indexación
    seleccion = st.radio(
        label="Selecciona la opción de indexación de embeddings deseada:",
        options={"CLIP", "M-CLIP"},
        index=0
    )
    index_selec = seleccion

    nombre_tabla = f"{nombre_tabla}_{index_selec}"

    if st.button("Crear tabla nueva"):
        #Eleccion nombre tabla nueva
        if nombre_tabla and nombre_tabla.rsplit('_',1)[0] not in [item[0] for item in st.session_state.vector_nombres_tabla]:
            st.info(f"Creando la tabla {nombre_tabla} con indexación {index_selec}...")
            st.session_state.nombre_tabla = nombre_tabla
            st.session_state.vector_nombres_tabla.append((nombre_tabla.rsplit('_',1)[0], index_selec))
            st.session_state.lancedb_manager.create_table(nombre_tabla,st.session_state.embedding_dim,True)
            st.info(f"Tabla '{nombre_tabla}' añadida")
        elif nombre_tabla and nombre_tabla.rsplit('_',1)[0] in [item[0] for item in st.session_state.vector_nombres_tabla]:
            st.info(f"La tabla '{nombre_tabla}' ya existe")
        else:
            st.info(f"Por favor, inserte el nombre que desea tener la nueva tabla")

        # Añadir a vector con los nombres de las tablas


        #st.write(tabla_nueva)

    if st.button("Mostrar tablas creadas"):
        st.info("Las tablas ya creadas son:")
        st.info(st.session_state.vector_nombres_tabla)

    nombre_tabla_eliminada = st.text_input("Escriba el nombre de la tabla que desea eliminiar")
    if st.button("Eliminar tabla:"):
        if not nombre_tabla_eliminada:
            st.info("Inserte el nombre de la tabla que desea eliminar")
        elif nombre_tabla_eliminada in [item[0] for item in st.session_state.vector_nombres_tabla]:
            for tabla_a_eliminar in st.session_state.vector_nombres_tabla:
                if nombre_tabla_eliminada == tabla_a_eliminar[0]:
                    modelo_a_eliminar = tabla_a_eliminar[1]
                    st.session_state.vector_nombres_tabla.remove(tabla_a_eliminar)

            #st.session_state.vector_nombres_tabla.remove(nombre_tabla_eliminada)
            st.session_state.lancedb_manager.eliminate_table(f"{nombre_tabla_eliminada}_{modelo_a_eliminar}")
            st.info(f"Se eliminó la tabla {nombre_tabla_eliminada}")
        else:
            st.info(f"La tabla {nombre_tabla_eliminada} no existe")

    


# --- Barra Lateral para Indexación ---
with tab1:
    
    col1_tab1, col2_tab1,col3_tab1,col4_tab1 = st.columns(4)

    with col1_tab1:
        st.header("📋 Seleccion de Tabla")
        
        #Mostrar nombres de las tablas en formato vertical
        seleccion = st.radio(
            label="Selecciona una tabla:",
            options=st.session_state.vector_nombres_tabla,
            index=0
        )
        LANCEDB_TABLE_NAME = f"{seleccion[0]}_{seleccion[1]}"
        LANCEDB_MODEL_TYPE = seleccion[1]



    with col2_tab1:
        st.header("⚙️ Indexar Fotos")
        st.info("Para indexar, especifica la ruta a tu carpeta de fotos. Las imágenes se procesarán y guardarán en la base de datos.")

        indexing_root_dir_input = st.text_input(
            "Ruta del Directorio Raíz de Fotos",
            value=st.session_state.indexing_root_dir_value,
            key="indexing_root_dir_text_input",
            help="Introduce la ruta absoluta de la carpeta que contiene tus fotos."
        )
        st.session_state.indexing_root_dir_value = indexing_root_dir_input

        if st.button("📁 Buscar Directorio"):
            if os.name in ['posix', 'nt']:
                st.warning("Se abrirá una nueva ventana para seleccionar el directorio. Por favor, ten paciencia.")
                selected_directory = choose_directory_dialog()
                if selected_directory:
                    st.session_state.indexing_root_dir_value = selected_directory
                    st.success(f"Directorio seleccionado: `{selected_directory}`")
                    st.rerun()
                else:
                    st.info("No se seleccionó ningún directorio.")
            else:
                st.error("La función 'Buscar Directorio' solo está disponible cuando se ejecuta Streamlit localmente en un sistema operativo de escritorio (Windows, macOS, Linux con GUI).")
                st.info("Si estás en un entorno de nube (ej. Colab), por favor, monta tu Google Drive y usa la ruta de Drive.")

        if st.button("🚀 Indexar Imágenes"):
            current_indexing_root_dir = st.session_state.indexing_root_dir_value

            if not os.path.isdir(current_indexing_root_dir):
                st.error(f"El directorio '{current_indexing_root_dir}' no es válido o no existe. Por favor, verifica la ruta.")
            else:
                with st.spinner("Buscando imágenes..."):
                    all_image_paths = find_image_files(current_indexing_root_dir)

                if not all_image_paths:
                    st.warning("No se encontraron imágenes en el directorio especificado. Revisa la ruta o añade imágenes.")
                else:
                    st.success(f"Encontradas {len(all_image_paths)} imágenes. Iniciando generación de embeddings e indexación...")
                
                    # --- Inicio de la barra de progreso ---
                    progress_text_status = st.empty() # Para mostrar texto de estado dinámico
                    progress_bar = st.progress(0, text="Progreso general...")

                    total_embeddings_collected = []
                    total_processed_paths = []
                
                    # Usar el generador para procesar lotes y actualizar la barra de progreso
                    for processed_count, total_count, batch_embeddings_raw, batch_paths_current_batch in st.session_state.clip_generator.generate_embeddings_batch_by_batch(
                        all_image_paths, batch_size=BATCH_SIZE_INDEXING
                    ):
                        if batch_embeddings_raw.size > 0: # Solo añadir si el lote no está vacío
                            total_embeddings_collected.append(batch_embeddings_raw)
                            total_processed_paths.extend(batch_paths_current_batch)

                        # Calcular y actualizar la barra de progreso
                        percentage = processed_count / total_count
                        progress_bar.progress(percentage, text=f"Generando embeddings: {processed_count}/{total_count} imágenes procesadas.")

                    progress_bar.empty() # Ocultar la barra de progreso cuando termina

                    if total_embeddings_collected:
                        # Normalizar TODOS los embeddings una vez que se han recolectado
                        final_embeddings = normalize(np.vstack(total_embeddings_collected))
                    
                        st.session_state.lancedb_manager.add_images_to_db(LANCEDB_TABLE_NAME, total_processed_paths, final_embeddings)
                        st.success(f"¡Indexación completada! Se añadieron {len(total_processed_paths)} imágenes a la base de datos.")
                    else:
                        st.error("No se pudieron generar embeddings para ninguna imagen. Revisa los archivos o los permisos.")

        if st.button("♻️ Eliminar Duplicados"):
            st.info("Comienzo del proceso para eliminar imagenes duplicadas")
            # 1) Crear segunda tabla provisional vacía
            st.session_state.tabla_nueva=st.session_state.lancedb_manager.create_table("LANCEDB_TABLE_NAME_NUEVA",st.session_state.embedding_dim,True)
            tabla_vieja=st.session_state.lancedb_manager.get_table(LANCEDB_TABLE_NAME)

            # 2) Comprobar si las imagenes de la tabla original se encuentran en la tabla nueva (distancia = 0), y si no es así pasarlas
            df_viejo = tabla_vieja.to_pandas() #Convertir tabla vieja en DataFrame
            st.info(f"Imágenes originales = {len(df_viejo)}")
            

            for i in range(len(df_viejo)):
                imagen_vieja = df_viejo.iloc[i]
#                st.info(imagen_vieja)
                YaEncuentra = False

                # Buscar si ya existe en tabla nueva
                resultados = st.session_state.tabla_nueva.search(imagen_vieja["embedding"]) \
                                        .metric("cosine") \
                                        .limit(1) \
                                        .to_pandas()
                if not resultados.empty and resultados.iloc[0]['_distance'] <= 0.00001:
                    YaEncuentra = True
#                    st.info("No añadir imagen")
#                else:
#                    st.info("Añadir imagen")


                
                #Si no existe, se añade
                if YaEncuentra == False:
                    st.session_state.tabla_nueva.add([{
                        "id":imagen_vieja["id"],
                        "path":imagen_vieja["path"],
                        "embedding":imagen_vieja["embedding"],
                        "indexed_at":imagen_vieja["indexed_at"]
                    }]
                    )

            st.success(f"Imagenes nuevas = {len(st.session_state.tabla_nueva)}")
            st.success(f"Imagenes eliminadas = {len(df_viejo) - len(st.session_state.tabla_nueva)}")
#            st.write(st.session_state.tabla_nueva)
                
            # 3) Eliminar tabla vieja y sustituirla por la nueva

            st.session_state.lancedb_manager.eliminate_table(LANCEDB_TABLE_NAME)
#            st.write("Borrado completado exitosamente")
            tabla = st.session_state.lancedb_manager.create_table(LANCEDB_TABLE_NAME,st.session_state.embedding_dim,True)
#            st.write(tabla)
            datos_a_copiar = st.session_state.tabla_nueva.to_arrow() #Obtención de los datos de la tabla nueva formato arrow
            tabla.add(datos_a_copiar)
#            st.write(st.session_state.tabla_nueva)
            st.write("Duplicados eliminados exitosamente")

            # 4) Eliminar tabla provisional
            #st.session_state.vector_nombres_tabla.remove("LANCEDB_TABLE_NAME_NUEVA")

            #st.session_state.vector_nombres_tabla.remove(nombre_tabla_eliminada)
            st.session_state.lancedb_manager.eliminate_table("LANCEDB_TABLE_NAME_NUEVA")




        if st.button("🗑️ Reiniciar Base de Datos"):
            st.info("Reiniciando proceso de reinicio")
            #st.session_state.lancedb_manager.create_table(LANCEDB_TABLE_NAME,st.session_state.embedding_dim,True)
#            db=lancedb.connect(lancedb_manager.py)
            with st.spinner("Reiniciando Base de Datos"):
                try:
                    st.session_state.lancedb_manager.eliminate_table(LANCEDB_TABLE_NAME)
                    st.session_state.lancedb_manager.create_table(LANCEDB_TABLE_NAME,st.session_state.embedding_dim,True)
                    print("Reinicio completado exitosamente")
                except Exception as e:
                    print(f"Error al intentar eliminar la tabla")


    with col3_tab1:
        st.subheader("✏️ Indexar documento con URLs")
        st.info("Para indexar, especifica la ruta a tu documento, que deberá contener las direcciones de las imágenes en cada línea. Las imágenes se procesarán y guardarán en la base de datos.")
        
        indexing_root_urls_input = st.text_input(
            "Ruta del documento Raíz de Fotos",
            value=st.session_state.indexing_root_urls_value,
            key="indexing_root_urls_text_input",
            help="Introduce la ruta absoluta del documento que contiene tus fotos."
        )
        st.session_state.indexing_root_urls_value = indexing_root_urls_input

        if st.button("📄 Buscar Documento"):
            if os.name in ['posix', 'nt']:
                st.warning("Se abrirá una nueva ventana para seleccionar el documento. Por favor, ten paciencia.")
                selected_document = choose_file_dialog() #Crear funcion
                if selected_document:
                    st.session_state.indexing_root_urls_value = selected_document
                    st.success(f"Documento seleccionado: `{selected_document}`")
                    st.rerun()
                else:
                    st.info("No se seleccionó ningún documento.")
            else:
                st.error("La función 'Buscar Documento' solo está disponible cuando se ejecuta Streamlit localmente en un sistema operativo de escritorio (Windows, macOS, Linux con GUI).")
                st.info("Si estás en un entorno de nube (ej. Colab), por favor, monta tu Google Drive y usa la ruta de Drive.")

        
        if st.button("🚀 Indexar Imágenes",key="indexar_desde_documento"):
            document_path = st.session_state.indexing_root_urls_value
            if not os.path.isfile(document_path):
                st.error(f"El documento '{document_path}' no existe o no es válido.")
            else:
                try:
                    # Leer URLs del documento
                    with open(document_path, 'r', encoding='utf-8') as f:
                        urls = [line.strip() for line in f.readlines() if line.strip()]
                    
                    if not urls:
                        st.warning("El documento está vacío o no contiene URLs válidas.")
                    else:
                        st.success(f"Encontradas {len(urls)} URLs. Iniciando descarga e indexación...")
                        
                        # Contadores para estadísticas
                        total_urls = len(urls)
                        successful_downloads = 0
                        failed_downloads = 0
                        non_image_urls = 0
                        processing_errors = 0
                        
                        # Listas para almacenar resultados
                        downloaded_image_paths = []
                        downloaded_images = []
                        
                        # --- Progreso de descarga ---
                        progress_bar_download = st.progress(0, text="Descargando imágenes...")
                        status_text = st.empty()
                        
                        # Descargar imágenes
                        for i, url in enumerate(urls):
                            try:
                                status_text.text(f"Descargando {i+1}/{total_urls}: {url[:50]}...")
                                
                                # Verificar si es una URL de imagen
                                if not any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']):
                                    non_image_urls += 1
                                    continue
                                
                                # Descargar imagen
                                response = requests.get(url, timeout=10, stream=True)
                                if response.status_code == 200:
                                    # Crear nombre de archivo temporal único
                                    ext = os.path.splitext(url)[1] or '.jpg'
                                    temp_path = f"imagenes/temp_downloaded_image_{i}{ext}"
                                    try:
                                    # Guardar imagen
                                        print(temp_path)
                                        with open(temp_path, 'wb') as img_file:
                                            for chunk in response.iter_content(1024):
                                                img_file.write(chunk)
                                    except:
                                        print(temp_path)    
                                    # Verificar que es una imagen válida
                                    try:
                                        img = Image.open(temp_path)
                                        img.verify()  # Verificar integridad
                                        downloaded_image_paths.append(temp_path)
                                        successful_downloads += 1
                                    except:
                                        os.remove(temp_path)
                                        failed_downloads += 1
                                else:
                                    failed_downloads += 1
                                    
                            except Exception as e:
                                failed_downloads += 1
                            
                            # Actualizar progreso
                            progress_bar_download.progress((i + 1) / total_urls)
                        
                        progress_bar_download.empty()
                        status_text.empty()
                        
                        # Mostrar estadísticas de descarga
                        #col1, col2, col3, col4 = st.columns(4)
                        #with col1:
                        st.info(f"URLs totales = {total_urls}")
                        #with col2:
                        st.info(f"Descargas exitosas = {successful_downloads}")
                        #with col3:
                        st.info(f"URLs no imágenes = {non_image_urls}")
                        #with col4:
                        st.info(f"Errores descarga = {failed_downloads}")
                        
                        # Procesar imágenes descargadas
                        if downloaded_image_paths:
                            st.info(f"Procesando {len(downloaded_image_paths)} imágenes descargadas...")
                            
                            # --- Progreso de procesamiento ---
                            progress_bar_process = st.progress(0, text="Generando embeddings...")
                            total_embeddings_collected = []
                            total_processed_paths = []
                            
                            st.success("Embeddings generados")

                            # Usar el generador para procesar lotes
                            for processed_count, total_count, batch_embeddings_raw, batch_paths_current_batch in st.session_state.clip_generator.generate_embeddings_batch_by_batch(
                                downloaded_image_paths, batch_size=BATCH_SIZE_INDEXING
                            ):
                                if batch_embeddings_raw.size > 0:
                                    total_embeddings_collected.append(batch_embeddings_raw)
                                    total_processed_paths.extend(batch_paths_current_batch)
                                
                                # Actualizar progreso
                                percentage = processed_count / len(downloaded_image_paths)
                                progress_bar_process.progress(percentage, text=f"Procesando: {processed_count}/{len(downloaded_image_paths)} imágenes")
                            
                            st.success("Lotes generados")

                            progress_bar_process.empty()
                            
                            if total_embeddings_collected:
                                # Normalizar embeddings
                                final_embeddings = normalize(np.vstack(total_embeddings_collected))
                                
                                # Añadir a la base de datos con URLs como metadata
                                metadata_list = [
                                    {"url": urls[i], "local_path": path} 
                                    for i, path in enumerate(total_processed_paths)
                                ]
                                
                                st.session_state.lancedb_manager.add_images_to_db(
                                    LANCEDB_TABLE_NAME, 
                                    total_processed_paths, 
                                    final_embeddings,
#                                    metadata_list=metadata_list
                                )
                                
                                st.success(f"✅ Indexación completada! {len(total_processed_paths)} imágenes añadidas desde URLs")
                                
                                # Limpiar archivos temporales
                                # for temp_path in downloaded_image_paths:
                                #     try:
                                #         os.remove(temp_path)
                                #     except:
                                #         pass
                                        
                            else:
                                st.error("No se pudieron generar embeddings para las imágenes descargadas.")
                        else:
                            st.warning("No se pudieron descargar imágenes válidas para procesar.")
                            
                except Exception as e:
                    st.error(f"Error procesando el documento: {str(e)}")






    with col4_tab1:
#    st.markdown("---")
        st.subheader("ℹ️ Información de la DB")
        try:
            st.write(LANCEDB_TABLE_NAME)
            table = st.session_state.lancedb_manager.get_table(LANCEDB_TABLE_NAME)
            if table:
                st.write(f"**Tabla:** `{LANCEDB_TABLE_NAME.rsplit('_',1)[0]}`")
                st.write(f"**Modelo:** `{LANCEDB_MODEL_TYPE}`")
                st.write(f"**Registros:** `{table.count_rows()}`")
                st.write(f"**Dimensión de Embedding:** `{st.session_state.embedding_dim}`")
            else:
                st.warning("La tabla de la base de datos aún no está inicializada o no existe.")
        except Exception as e:
            st.error(f"Error al obtener información de la DB: {e}")

# --- Sección Principal para Búsqueda ---
with tab2:
    st.header("🔍 Buscar Fotos")

    col1_tab2, col2_tab2 = st.columns(2)
    with col1_tab2:
        search_method = st.radio(
            "Selecciona tu método de búsqueda:",
            ("Búsqueda por Texto", "Búsqueda por Imagen", "Búsqueda por Imagen-Texto", "Búsqueda Texto sobre Imagen"),
            index=0,
            key="search_method_radio"
        )
    with col2_tab2:
        #Mostrar opciones de indexación
        seleccion = st.radio(
            label="Selecciona la tabla de embeddings que desea ser indexada:",
            options=st.session_state.vector_nombres_tabla,
            index=0,
            key="embedding_option_radio"
        )
        LANCEDB_TABLE_NAME = f"{seleccion[0]}_{seleccion[1]}"
        LANCEDB_MODEL_TYPE = seleccion[1]
        st.session_state.tabla_seleccionada ={
            "nombre": seleccion[0],
            "model_type":seleccion[1]
        } 

    search_limit = st.slider(
        "Número de resultados a mostrar:",
        min_value=1, max_value=50, value=SEARCH_LIMIT_DEFAULT, step=1, key="resultados_busqueda"
    )

    query_embedding = None
    query_label = ""

    if search_method == "Búsqueda por Texto":
        query_text = st.text_input("Describe la imagen que buscas (ej. 'un perro jugando en el parque', 'una persona sonriendo', 'comida italiana')", "")
        if query_text:
            query_embedding = st.session_state.clip_generator.get_text_embedding(query_text)
            query_label = f"Texto: '{query_text}'"
            
    elif search_method == "Búsqueda por Imagen":
        uploaded_file = st.file_uploader("Sube una imagen similar a lo que buscas", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding = None


    ### Opción 1: primero busca sobre una imagen y un texto
    elif search_method == "Búsqueda por Imagen-Texto":
        uploaded_file = st.file_uploader("Sube una imagen similar a lo que buscas", type=["jpg", "jpeg", "png", "webp"])
        query_text = st.text_input("Describe la imagen que buscas (ej. 'un perro jugando en el parque', 'una persona sonriendo', 'comida italiana')", "")

        if (uploaded_file is not None) and (query_text):
            try:

                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)

                # Se usa el generador con la imagen
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield

                # Se calcula el embedding de la imagen
                
                if batch_embeddings_raw.ndim == 1:	#Con el if nos aseguramos que el embedding de la imagen sea 2D
                    batch_embeddings_raw = batch_embeddings_raw[np.newaxis, :] 	# Convert (D,) to (1, D)
                elif batch_embeddings_raw.ndim == 3 and batch_embeddings_raw.shape[0] == 1 and batch_embeddings_raw.shape[1] == 1:
                    # This specific case (1, 1, D) is often the cause of the 3D error
                    batch_embeddings_raw = batch_embeddings_raw.reshape(1, -1)
                query_embedding_imagen = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
    #            query_label_imagen = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)

                # Se calcula el embedding del texto
                query_embedding_texto_raw = st.session_state.clip_generator.get_text_embedding(query_text)
                
                if query_embedding_texto_raw.ndim == 1:	#Con el if nos aseguramos que el embedding del texto sea 2D
                    query_embedding_texto_raw = query_embedding_texto_raw[np.newaxis, :] 	# Convert (D,) to (1, D)
                elif query_embedding_texto_raw.ndim == 3 and query_embedding_texto_raw.shape[0] == 1 and query_embedding_texto_raw.shape[1] == 1:
                    query_embedding_texto_raw = query_embedding_texto_raw.reshape(1, -1)
                    
                query_embedding_texto = normalize(query_embedding_texto_raw)[0]                
    #            query_label_texto = f"Texto: '{query_text}'"
                

                #Ajuste de pesos
    #            weight_image = 0.5
    #            weight_text = 0.5
    #            query_embedding_fused = (weight_image * query_embedding_imagen) + (weight_text * query_embedding_texto)
    #            if query_embedding_fused.ndim == 1:
    #                query_embedding_fused = query_embedding_fused[np.newaxis, :]
    #            query_embedding = normalize(query_embedding_fused)[0] # Normalize the final fused embedding
                
                query_label = f"Imagen: '{uploaded_file.name}' + Texto: '{query_text}'"
                
                #Se realizarán dos búsquedas separadas, una por cada tipo, y luego se combinarán los resultados
                # 1) Busqueda con el embedding de la imagen
                image_search_results = st.session_state.lancedb_manager.search_images(
                    LANCEDB_TABLE_NAME, query_embedding_imagen, limit=search_limit * 2 # Get more results to allow for better merging
                )
                
                # 2) Busqueda con el embedding del texto
                text_search_results = st.session_state.lancedb_manager.search_images(
                    LANCEDB_TABLE_NAME, query_embedding_texto, limit=search_limit * 2 # Get more results
                )
                
                # 3) Se combinan los resultados
                combined_results_map = {} # Key: image_path, Value: {'_distance_image': val, '_distance_text': val, 'original_data': {...}}

                # Resultados del embeding de la imagen
                for res in image_search_results:
                    path = res['path']
                    if path not in combined_results_map:
                        combined_results_map[path] = {
                            '_distance_image': res['_distance'],
                            '_distance_text': float('inf'), # Se inicializa la distancia a infinito
                            'path': path, 
                            'original_res': res 
                        }
                    else:
                        # Incluir imagen en el path
                        combined_results_map[path]['_distance_image'] = min(combined_results_map[path]['_distance_image'], res['_distance'])

                # Repetir con los resultados del texto
                for res in text_search_results:
                    path = res['path']
                    if path not in combined_results_map:
                        combined_results_map[path] = {
                            '_distance_image': float('inf'), 
                            '_distance_text': res['_distance'],
                            'path': path,
                            'original_res': res
                        }
                    else:
                        combined_results_map[path]['_distance_text'] = min(combined_results_map[path]['_distance_text'], res['_distance'])

                # Convertie mapa a lista y calcular un resultado combinado
                # Razón: Calcular una métrica combinada para clasificar los resultados fusionados.
                # El promedio de distancias es una forma sencilla y efectiva. Cuanto menor la distancia, mejor.
                final_fused_results = []
                for path, data in combined_results_map.items():
                    # Se evita dividir entre 0 o infinito
                    score_image = data['_distance_image'] if data['_distance_image'] != float('inf') else 2.5 # Max distance for no match
                    score_text = data['_distance_text'] if data['_distance_text'] != float('inf') else 2.5 # Max distance for no match

                    # Calculo de distancia haciendo la media de las distancias de cada enbeding
                    combined_score = (score_image + score_text) / 2.0
                    
                    # Actualizar diccionario
                    result_for_display = data['original_res']
                    result_for_display['_distance'] = combined_score # Overwrite _distance for display
                    final_fused_results.append(result_for_display)

                final_fused_results.sort(key=lambda x: x['_distance'])

                query_embedding = "fused_query_placeholder" 
                search_results = final_fused_results[:search_limit] # N primeros resultados
                
                st.session_state.current_search_results = final_fused_results[:search_limit]
                query_embedding = None

                # Se combinan los dos embeddings en uno solo
                #Intento con concatenación
    #            query_embedding = np.concatenate((query_embedding_imagen, query_embedding_texto), axis=1)
    #            query_label = f"Imagen: '{uploaded_file.name}' + Texto: '{query_text}'"

                #Intento con producto escalar
    #            query_embedding = np.dot((query_embedding_imagen, query_embedding_texto.T))	#Producto escalar
    #            query_embedding = query_embedding.item() 	#Dimension 1
    #            query_label = f"Imagen: '{uploaded_file.name}' + Texto: '{query_text}'"



            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding = None

    ### Opción 2: primero busca una imagen y ordena según las distancias del embedding del texto
    elif search_method == "Búsqueda Texto sobre Imagen":
        uploaded_file = st.file_uploader("Sube una imagen similar a lo que buscas", type=["jpg", "jpeg", "png", "webp"])
        query_text = st.text_input("Describe la imagen que buscas (ej. 'un perro jugando en el parque', 'una persona sonriendo', 'comida italiana')", "")
        
        query_embedding_imagen = None
        query_embedding_texto = None

        if (uploaded_file is not None) and (query_text):
            try:

                #1. Cálculo embeddings imagen
                
                #Imagen
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator)
                
                if batch_embeddings_raw.ndim == 1:
                    batch_embeddings_raw = batch_embeddings_raw[np.newaxis, :]
                elif batch_embeddings_raw.ndim == 3 and batch_embeddings_raw.shape[0] == 1 and batch_embeddings_raw.shape[1] == 1:
                    batch_embeddings_raw = batch_embeddings_raw.reshape(1, -1)
                    
                query_embedding_imagen = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                #query_label = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)
                
                #Texto
                query_embedding_texto_raw = st.session_state.clip_generator.get_text_embedding(query_text)
                
                if query_embedding_texto_raw.ndim == 1:
                    query_embedding_texto_raw = query_embedding_texto_raw[np.newaxis, :]
                elif query_embedding_texto_raw.ndim == 3 and query_embedding_texto_raw.shape[0] == 1 and query_embedding_texto_raw.shape[1] == 1:
                    query_embedding_texto_raw = query_embedding_texto_raw.reshape(1, -1)
                
                query_embedding_texto = normalize(query_embedding_texto_raw)[0]
                
                query_label = f"Imagen base: '{uploaded_file.name}' y re-ranking por Texto: '{query_text}'"
                
                #2. Busqueda con la imagen
                
                initial_image_search_results = st.session_state.lancedb_manager.search_images(
                    LANCEDB_TABLE_NAME, query_embedding_imagen, limit=search_limit * 5 # Busqueda inicial mayor para tener más candidatos a eliminar
                )
                
                if not initial_image_search_results:
                    st.info("No se encontraron resultados iniciales de imagen.")
                    st.session_state.current_search_results = []
                    st.stop()
                    
                paths_to_rerank = [res['path'] for res in initial_image_search_results] # Se guardan los embeddings de los resultados de la imagen en un vector para facilitar la comparación con la distancia al texto
                
                #3. Cálculo de la distancia entre el embedding de cada imagen obtenida y el del texto introducido
                
                reranked_results = []
                
                for res_item in initial_image_search_results:
                    image_vector = res_item.get('embedding') # Obtención de la imagen del vector
                    if image_vector is not None:
                        # Calculate cosine distance (or similarity)
                        # For cosine distance, it's 1 - cosine_similarity.
                        # cosine_similarity = np.dot(a, b) / (norm(a) * norm(b))
                        # Since embeddings are already normalized, norm is 1. So, dot product is similarity.
                        # Distance = 1 - similarity.
                        
                        # Comprobación de que las dimensiones sean 1D
                        image_vector = np.asarray(image_vector).flatten()
                        text_vector = query_embedding_texto.flatten()

                        # Check for non-zero vectors to avoid NaN in similarity calculation
                        if np.linalg.norm(image_vector) > 1e-9 and np.linalg.norm(text_vector) > 1e-9:
                            text_distance = 1 - np.dot(image_vector, text_vector) # Cosine distance
                            print(text_distance)
                        else:
                            text_distance = 2.0 # Max distance if a vector is zero (no match)

                        # Se actualiza la distancia coseno
                        res_item['_distance'] = text_distance
                        reranked_results.append(res_item)
                    else:
                        st.warning(f"No se encontró el embedding para la imagen: {res_item['path']}")

                # Ordenar los resultados por la nueva distancia de texto
                # Razón: Los resultados finales deben reflejar la relevancia textual
                # dentro del subconjunto de imágenes obtenidas por la similitud visual.
                reranked_results.sort(key=lambda x: x['_distance'])
                
                st.session_state.current_search_results = reranked_results[:search_limit]




            except Exception as e:
                st.error(f"Error al procesar Texto sobre Imagen: {e}")
                st.session_state.current_search_results = []



    if st.button("🔍 Buscar", key = "Busqueda"):
        #Cargar el modelo correspondiente
        load_clip_model(LANCEDB_MODEL_TYPE)

        # Asegurarse de que la tabla exista antes de intentar buscar
        table = st.session_state.lancedb_manager.get_table(LANCEDB_TABLE_NAME)
        if search_method == "Búsqueda por Imagen-Texto" and 'current_search_results' in st.session_state:
            search_results = st.session_state.current_search_results
            display_image_results(search_results)
            # Limpiar los resultados temporales para la siguiente búsqueda
            del st.session_state.current_search_results 
        elif search_method == "Búsqueda Texto sobre Imagen" and 'current_search_results' in st.session_state:
            search_results = st.session_state.current_search_results
            display_image_results(search_results)
            # Limpiar los resultados temporales para la siguiente búsqueda
            del st.session_state.current_search_results 
        
        
        elif query_embedding is not None and table is not None and table.count_rows() > 0:
            with st.spinner(f"Buscando imágenes similares a {query_label}..."):
                search_results = st.session_state.lancedb_manager.search_images(
                    LANCEDB_TABLE_NAME, query_embedding, limit=search_limit
                )
            display_image_results(search_results)
        elif query_embedding is None:
            st.warning("Por favor, introduce un texto o sube una imagen para buscar.")
        elif table is None:
            st.error("La base de datos no está inicializada. Por favor, indexa algunas imágenes primero.")
        else: # table.count_rows() == 0
            st.warning("La base de datos está vacía. Por favor, indexa algunas imágenes primero.")

# --- Sección para realizar operaciones algebráicas con embeddings
with tab3:
    col1_tab3, col2_tab3 = st.columns(2)
    with col1_tab3:
        type_operation = st.radio(
            "Selecciona la operación que se desea utilizar:",
            ("Suma de Textos", "Resta de Textos", "Suma de Imágenes", "Resta de Imágenes", "Suma de Texto e Imagen", "Resta Texto - Imagen", "Resta Imagen - Texto", "Operaciones Combinadas"),
            index=0,
            key="type_operation_radio"
        )
    with col2_tab3:
        #Mostrar opciones de indexación
        seleccion = st.radio(
            label="Selecciona la tabla de embeddings que desea ser indexada:",
            options=st.session_state.vector_nombres_tabla,
            index=0,
            key="table_option_radio"
        )
        LANCEDB_TABLE_NAME = f"{seleccion[0]}_{seleccion[1]}"
        LANCEDB_MODEL_TYPE = seleccion[1]
        st.session_state.tabla_seleccionada ={
            "nombre": seleccion[0],
            "model_type":seleccion[1]
        } 

    operaciones_a_realizar = st.slider(
        "Número de resultados a mostrar:",
        min_value=1, max_value=50, value=SEARCH_LIMIT_DEFAULT, step=1, key="resultados_operaciones"
    )

    query_embedding = None
    query_label = ""
        
    if type_operation == "Suma de Textos":
        query_text_1 = st.text_input("Introduzca el primer sumando","")
        query_text_2 = st.text_input("Introduzca el segundo sumando","")

        # 1. Procesamiento de Embeddings
        if query_text_1:
            query_embedding_1 = st.session_state.clip_generator.get_text_embedding(query_text_1)
            query_label_1 = f"Texto: '{query_text_1}'"
        if query_text_2:
            query_embedding_2 = st.session_state.clip_generator.get_text_embedding(query_text_2)
            query_label_2 = f"Texto: '{query_text_2}'"

        # 2. Suma
        if  query_text_1 and query_text_2:
            query_embedding_suma = query_embedding_1 + query_embedding_2
            query_label = f"Texto: '{query_text_1} + {query_text_2}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding_suma)[0]

    elif type_operation == "Resta de Textos":
        query_text_1 = st.text_input("Introduzca el embedding que buscas","")
        query_text_2 = st.text_input("Introduzca el embedding que deseas evitar","")

        # 1. Procesamiento de Embeddings
        if query_text_1:
            query_embedding_1 = st.session_state.clip_generator.get_text_embedding(query_text_1)
            query_label_1 = f"Texto: '{query_text_1}'"
        if query_text_2:
            query_embedding_2 = st.session_state.clip_generator.get_text_embedding(query_text_2)
            query_label_2 = f"Texto: '{query_text_2}'"

        # 2. Resta
        if  query_text_1 and query_text_2:
            query_embedding_resta = query_embedding_1 - query_embedding_2
            query_label = f"Texto: '{query_text_1} - {query_text_2}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding_resta)[0]


    elif type_operation == "Suma de Imágenes":
        uploaded_file_1 = st.file_uploader("Sube el primer sumando", type=["jpg", "jpeg", "png", "webp"])
        uploaded_file_2 = st.file_uploader("Sube el segundo sumando", type=["jpg", "jpeg", "png", "webp"])

        # 1. Procesamiento de Embeddings
        if uploaded_file_1 is not None:
            try:
                image = Image.open(uploaded_file_1).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_1 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_1 = f"Imagen: '{uploaded_file_1.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_1 = None
        
        if uploaded_file_2 is not None:
            try:
                image = Image.open(uploaded_file_2).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_2 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_2 = f"Imagen: '{uploaded_file_2.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_2 = None

        # 2. Suma
        if  uploaded_file_1 and uploaded_file_2:
            query_embedding = query_embedding_1 + query_embedding_2
            query_label = f"Imagen: '{uploaded_file_1} - {uploaded_file_2}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]


    elif type_operation == "Resta de Imágenes":
        uploaded_file_1 = st.file_uploader("Introduzca el embedding que buscas", type=["jpg", "jpeg", "png", "webp"])
        uploaded_file_2 = st.file_uploader("Sube el sustraendo", type=["jpg", "jpeg", "png", "webp"])

        # 1. Procesamiento de Embeddings
        if uploaded_file_1 is not None:
            try:
                image = Image.open(uploaded_file_1).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_1 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_1 = f"Imagen: '{uploaded_file_1.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_1 = None
        
        if uploaded_file_2 is not None:
            try:
                image = Image.open(uploaded_file_2).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_2 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_2 = f"Imagen: '{uploaded_file_2.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_2 = None

        # 2. Resta
        if  uploaded_file_1 and uploaded_file_2:
            query_embedding = query_embedding_1 - query_embedding_2
            query_label = f"Imagen: '{uploaded_file_1} - {uploaded_file_2}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]

    elif type_operation == "Suma de Texto e Imagen":
        query_text = st.text_input("Introduzca el primer sumando","")
        uploaded_file = st.file_uploader("Introduzca el embedding que deseas evitar", type=["jpg", "jpeg", "png", "webp"])

        # 1. Procesamiento de Embeddings
        if query_text:
            query_embedding_1 = st.session_state.clip_generator.get_text_embedding(query_text)
            query_label_1 = f"Texto: '{query_text}'"

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_2 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_2 = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_2 = None

        # 2. Suma 
        if  query_text and uploaded_file:
            query_embedding = query_embedding_1 + query_embedding_2
            query_label = f"Imagen: '{query_text} + {uploaded_file}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]

    elif type_operation == "Resta Texto - Imagen":
        query_text = st.text_input("Introduzca el embedding que buscas","")
        uploaded_file = st.file_uploader("Introduzca el embedding que deseas evitar", type=["jpg", "jpeg", "png", "webp"])
        alpha = st.slider("Seleccione el peso de la imagen a restar:", 0.1, 0.5, 0.25, key="alpha_imagen_texto")

        # 1. Procesamiento de Embeddings
        if query_text:
            query_embedding_1 = st.session_state.clip_generator.get_text_embedding(query_text)
#            query_embedding_1 = normalize(query_embedding_1)[0]
            query_label_1 = f"Texto: '{query_text}'"

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_2 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_2 = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_2 = None
        
        # 2. Resta 
        if  query_text and uploaded_file:
            query_embedding = query_embedding_1 - (alpha * query_embedding_2)
            query_label = f"Imagen: '{query_text} - {uploaded_file}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]
        
    elif type_operation == "Resta Imagen - Texto":
        uploaded_file = st.file_uploader("Introduzca el embedding que buscas", type=["jpg", "jpeg", "png", "webp"])
        query_text = st.text_input("Introduzca el embedding que deseas evitar","")
        alpha = st.slider("Seleccione el peso del texto a restar:", 0.1, 5.0, 2.5, key="alpha_texto_imagen")

        # 1. Procesamiento de Embeddings
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_1 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_1 = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_1 = None

        if query_text:
            query_embedding_2 = st.session_state.clip_generator.get_text_embedding(query_text)
            query_label_2 = f"Texto: '{query_text}'"
        
        # 2. Resta 
        if  query_text and uploaded_file:
            query_embedding = query_embedding_1 - (alpha * query_embedding_2)
            query_label = f"Imagen: '{uploaded_file} - {query_text}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]
    
    elif type_operation == "Operaciones Combinadas":
        uploaded_file = st.file_uploader("Introduzca la imagen original",type=["jpg", "jpeg", "png", "webp"])
        query_text_suma = st.text_input("Introduzca el texto que deseas sumar","")
        beta = st.slider("Seleccione el peso del texto a sumar:", 0.1, 5.0, 2.5, key="beta_combinadas")
        query_text_resta = st.text_input("Introduzca el texto que deseas restar","")
        alpha = st.slider("Seleccione el peso del texto a restar:", 0.1, 5.0, 2.5, key="alpha_combinadas")

        # 1. Procesamiento de Embeddings
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Imagen de consulta", width=200)
                # Guardar la imagen temporalmente para generate_embeddings_batch_by_batch
                temp_img_path = "temp_query_image.jpg"
                image.save(temp_img_path)
                
                # Usamos el generador, pero solo con una imagen y tomamos el primer (y único) resultado
                single_embedding_generator = st.session_state.clip_generator.generate_embeddings_batch_by_batch([temp_img_path], batch_size=1)
                _, _, batch_embeddings_raw, _ = next(single_embedding_generator) # Obtener el primer yield
                
                query_embedding_1 = normalize(batch_embeddings_raw)[0] # Normalizar aquí y tomar el primer (y único) embedding
                query_label_1 = f"Imagen: '{uploaded_file.name}'"
                os.remove(temp_img_path)
                
            except Exception as e:
                st.error(f"Error al procesar la imagen subida: {e}")
                query_embedding_1 = None

        if query_text_suma:
            query_embedding_suma = st.session_state.clip_generator.get_text_embedding(query_text_suma)
            query_label_suma = f"Texto: '{query_text_suma}'"

        if query_text_resta:
            query_embedding_resta = st.session_state.clip_generator.get_text_embedding(query_text_resta)
            query_label_resta = f"Texto: '{query_text_resta}'"
        
        # 2. Operacion 
        if  query_text_suma and query_text_resta and uploaded_file:
            query_embedding = query_embedding_1 + (beta * query_embedding_suma) - (alpha * query_embedding_resta)
            query_label = f"Imagen: '{uploaded_file} + {query_text_suma}- {query_text_resta}'"

        # 3. Normalización
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]


    if st.button("🔍 Calcular", key = "Calculo"):
        #Cargar el modelo correspondiente
        load_clip_model(LANCEDB_MODEL_TYPE) 
        
        # Asegurarse de que la tabla exista antes de intentar buscar
        table = st.session_state.lancedb_manager.get_table(LANCEDB_TABLE_NAME)

        if query_embedding is not None and table is not None and table.count_rows() > 0:
            with st.spinner(f"Buscando imágenes similares a {query_label}..."):
                search_results = st.session_state.lancedb_manager.search_images(
                    LANCEDB_TABLE_NAME, query_embedding, limit=operaciones_a_realizar
                )
            display_image_results(search_results)
        elif query_embedding is None:
            st.warning("Por favor, introduce un texto o sube una imagen para buscar.")
        elif table is None:
            st.error("La base de datos no está inicializada. Por favor, indexa algunas imágenes primero.")
        else: # table.count_rows() == 0
            st.warning("La base de datos está vacía. Por favor, indexa algunas imágenes primero.")


st.markdown("---")
st.markdown("Desarrollado con ❤️ y Python por tu asistente de IA.")