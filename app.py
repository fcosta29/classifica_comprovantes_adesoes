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
    # Converte a imagem para escala de cinza
    imagem_gray = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

    # Equalização de histograma para melhorar o contraste
    imagem_gray = cv2.equalizeHist(imagem_gray)
    
    # Suavização para reduzir ruído
    blur = cv2.GaussianBlur(imagem_gray, (5, 5), 0)

    # Detecta características usando ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imagem_gray, None)

    # Cria uma imagem padrão do documento (idealmente, deve ter uma imagem "template" do documento)
    imagem_template = cv2.imread('template_documento.jpg', cv2.IMREAD_GRAYSCALE)  # Substitua com a imagem do documento padrão
    kp2, des2 = orb.detectAndCompute(imagem_template, None)

    # Usando o BFMatcher para encontrar correspondências
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Ordena as correspondências com base na distância
    matches = sorted(matches, key = lambda x:x.distance)

    # Usa as correspondências melhores para encontrar os pontos-chave
    pontos1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pontos2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Homografia para alinhar a imagem com o template do documento
    M, mask = cv2.findHomography(pontos1, pontos2, cv2.RANSAC, 5.0)

    if M is not None:
        # Aplica a transformação de perspectiva para corrigir a imagem
        h, w, _ = imagem_rgb.shape
        imagem_corrigida = cv2.warpPerspective(imagem_rgb, M, (w, h))
        
        # Detecta os contornos da imagem corrigida
        imagem_gray_corrigida = cv2.cvtColor(imagem_corrigida, cv2.COLOR_RGB2GRAY)
        _, imagem_binaria = cv2.threshold(imagem_gray_corrigida, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maior_area = 0
        contorno_documento = None

        for contorno in contornos:
            peri = cv2.arcLength(contorno, True)
            aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
            if len(aprox) == 4:
                area = cv2.contourArea(aprox)
                if area > maior_area:
                    maior_area = area
                    contorno_documento = aprox

        if contorno_documento is not None:
            return contorno_documento, imagem_corrigida
    return None, imagem_rgb

def recorte_documento(imagem, pontos):
    pontos = ordenar_pontos(pontos)
    (tl, tr, br, bl) = pontos

    larguraA = np.linalg.norm(br - bl)
    larguraB = np.linalg.norm(tr - tl)
    largura = max(int(larguraA), int(larguraB))

    alturaA = np.linalg.norm(tr - br)
    alturaB = np.linalg.norm(tl - bl)
    altura = max(int(alturaA), int(alturaB))

    destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pontos, destino)
    warped = cv2.warpPerspective(imagem, M, (largura, altura))
    return warped

def ordenar_pontos(pontos):
    pontos = pontos.reshape(4, 2)
    ret = np.zeros((4, 2), dtype="float32")
    s = pontos.sum(axis=1)
    ret[0] = pontos[np.argmin(s)]
    ret[2] = pontos[np.argmax(s)]
    diff = np.diff(pontos, axis=1)
    ret[1] = pontos[np.argmin(diff)]
    ret[3] = pontos[np.argmax(diff)]
    return ret

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