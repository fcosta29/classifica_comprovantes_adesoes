import streamlit as st
import gdown

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

def main():
    
    st.set_page_config(
    page_title="Classifica Comprovantes de adesão",
    )

    st.write("# Classifica comprovantes de adesão!")

    #carrega modelo
    interpreter = carrega_modelo()
    #carrega imagem
    #image, image_bytes = carrega_imagem()
    #st.write("Bytes da imagem")
    #st.write(image_bytes)
    #classifica
    #if image is not None:
        #previsao(interpreter,image)
        #Valida
        #valida_imagem_duplicada(image_bytes)

if __name__ == "__main__":
    main()