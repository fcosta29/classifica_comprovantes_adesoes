import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import io
import numpy as np
import pandas as pd
import plotly.express as px
import cv2

@st.cache_resource

def carrega_modelo():
          #https://drive.google.com/file/d/1jxwhxLYwmuSNOCLgQ8h46MHpyDpPeQ9o/view?usp=drive_link
          #https://drive.google.com/file/d/1HgKAh7KitA5d0ArlvRRE55WzvDN_hddd/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1HgKAh7KitA5d0ArlvRRE55WzvDN_hddd'
    output = 'modelo_guia_comprovantes_quantizado16bits.tflite'
    
    gdown.download(url, output)

    interpreter = tf.lite.Interpreter(model_path=output)
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Escolha um comprovante', type=['jpg','jpeg','png']) 

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        #image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        #st.image(image_pil, caption='Imagem Original', use_column_width=True)
        #st.success('Imagem foi carregada com sucesso')

        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        image = image.resize((520, 112))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image, image_data

        # imagem_np = np.array(image_pil)
        #contorno, imagem_corrigida = detectar_documento(imagem_np)

        #if contorno is not None:
            # Recorta o documento da imagem corrigida
            #documento_crop = recorte_documento(imagem_corrigida, contorno)
            #st.image(documento_crop, caption='Documento Detectado', use_column_width=True)

            # Redimensionar para o modelo
            #documento_model = cv2.resize(documento_crop, (520, 112))
            #documento_model = documento_model.astype(np.float32) / 255.0
            #documento_model = np.expand_dims(documento_model, axis=0)

            #return documento_model, image_data
        #else:
            #st.warning("Documento não encontrado. Verifique iluminação ou ângulo.")
            #return None, None

    return None, None

def previsao(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #st.write("Detalhes de Entrada (Input Details):")
    #st.json(input_details)

    #st.write("Detalhes de Saída (Output Details):")
    #st.json(output_details)
    
    interpreter.set_tensor(input_details[0]['index'],image) 
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['comprovantes_assinados','comprovantes_nao_assinados']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    #st.write("Probabilidades por Classe:")
    #st.json(df.set_index('classes')['probabilidades (%)'].round(2).to_dict())
    
    
    fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Comprovantes')
    
    st.plotly_chart(fig)

def main():

    st.set_page_config(
    page_title="Classifica Comprovantes de adesão",
    )

    st.write("# Classifica comprovantes de adesão!")

    #carrega modelo
    interpreter = carrega_modelo()
    #carrega imagem
    image, image_bytes = carrega_imagem()
    #st.write("Bytes da imagem")
    #st.write(image_bytes)
    #classifica
    if image is not None:
       previsao(interpreter,image)
        
        #Valida
        #valida_imagem_duplicada(image_bytes)

if __name__ == "__main__":
    main()