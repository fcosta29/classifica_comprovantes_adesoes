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
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        st.image(image_pil, caption='Imagem Original', use_column_width=True)
        st.success('Imagem foi carregada com sucesso')

        imagem_np = np.array(image_pil)
        contorno, imagem_corrigida = detectar_documento(imagem_np)

        if contorno is not None:
            # Recorta o documento da imagem corrigida
            documento_crop = recorte_documento(imagem_corrigida, contorno)
            st.image(documento_crop, caption='Documento Detectado', use_column_width=True)

            # Redimensionar para o modelo
            documento_model = cv2.resize(documento_crop, (520, 112))
            documento_model = documento_model.astype(np.float32) / 255.0
            documento_model = np.expand_dims(documento_model, axis=0)

            return documento_model, image_data
        else:
            st.warning("Documento não encontrado. Verifique iluminação ou ângulo.")
            return None, None

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

def detectar_documento(imagem_rgb):
    imagem = imagem_rgb.copy()
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    imagem_blur = cv2.GaussianBlur(imagem_gray, (5, 5), 0)
    imagem_edges = cv2.Canny(imagem_blur, 75, 200)

    contornos, _ = cv2.findContours(imagem_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    altura_img, largura_img = imagem.shape[:2]
    area_img = altura_img * largura_img

    melhor_contorno = None
    melhor_area = 0

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

        if len(aprox) != 4:
            continue

        area = cv2.contourArea(aprox)

        # Considerar documentos médios (por ex: entre 10% e 60% da imagem)
        if area < area_img * 0.1 or area > area_img * 0.6:
            continue

        pontos = aprox.reshape(4, 2)
        pontos_ordenados = ordenar_pontos(pontos)

        # Verifica proporção (largura vs altura) entre 1:2 e 2:1 — genérico
        (tl, tr, br, bl) = pontos_ordenados
        largura = np.linalg.norm(tr - tl)
        altura = np.linalg.norm(bl - tl)
        proporcao = altura / largura if largura > 0 else 0

        if 0.5 < proporcao < 2.0:  # permite formatos mais flexíveis
            if area > melhor_area:
                melhor_area = area
                melhor_contorno = pontos_ordenados

    if melhor_contorno is not None:
        return melhor_contorno, imagem_rgb

    return None, None

def recorte_documento(imagem, pontos):
    largura_doc = 800
    altura_doc = 1000

    destino = np.array([
        [0, 0],
        [largura_doc - 1, 0],
        [largura_doc - 1, altura_doc - 1],
        [0, altura_doc - 1]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(pontos, destino)
    imagem_transformada = cv2.warpPerspective(imagem, matriz, (largura_doc, altura_doc))
    return imagem_transformada

def ordenar_pontos(pontos):
    pontos = np.array(pontos, dtype="float32")
    soma = pontos.sum(axis=1)
    diff = np.diff(pontos, axis=1)

    topo_esquerdo = pontos[np.argmin(soma)]
    base_direita = pontos[np.argmax(soma)]
    topo_direito = pontos[np.argmin(diff)]
    base_esquerda = pontos[np.argmax(diff)]

    return np.array([topo_esquerdo, topo_direito, base_direita, base_esquerda], dtype="float32")

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