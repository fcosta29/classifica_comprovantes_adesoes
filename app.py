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
    imagem_gray = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

    # Detector ORB
    orb = cv2.ORB_create()

    # Detecta keypoints e descritores da imagem de entrada
    kp1, des1 = orb.detectAndCompute(imagem_gray, None)

    # Carrega template do documento em escala de cinza
    imagem_template = cv2.imread('template_documento.jpg', cv2.IMREAD_GRAYSCALE)
    if imagem_template is None:
        st.error("Erro: o template do documento não foi encontrado.")
        return None, imagem_rgb

    kp2, des2 = orb.detectAndCompute(imagem_template, None)

    # Verificações para evitar erro do OpenCV
    if des1 is None or des2 is None:
        st.warning("Não foi possível detectar características suficientes. Verifique a imagem ou o template.")
        return None, imagem_rgb

    if des1.shape[1] != des2.shape[1]:
        st.warning("Descritores incompatíveis entre imagem e template.")
        return None, imagem_rgb

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        st.warning("Correspondências insuficientes. Documento pode não estar visível.")
        return None, imagem_rgb

    # Ordena e extrai os melhores matches
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)

    # Estima homografia
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        st.warning("Não foi possível estimar a transformação do documento.")
        return None, imagem_rgb

    h, w = imagem_template.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Desenha a região detectada na imagem original
    imagem_resultado = imagem_rgb.copy()
    cv2.polylines(imagem_resultado, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Recorta a região do documento
    try:
        recorte = cv2.warpPerspective(imagem_rgb, M, (w, h))
    except:
        st.warning("Erro ao recortar o documento.")
        return None, imagem_resultado

    # Exibe resultado
    st.image(recorte, caption="Documento detectado", use_column_width=True)
    return recorte, imagem_resultado

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