# app.py
import gradio as gr
from main import predict_image

# Função de callback

def run(prediction_image, threshold):
    """
    Executa a inferência dividindo a imagem e retorna:
    - lista de imagens segmentadas (Gallery)
    - lista de contagens por parte (JSON)
    - total de ovos (Number)
    """
    results, total = predict_image(prediction_image, threshold)
    imgs   = [im for im, _ in results]
    counts = [cnt for _, cnt in results]
    return imgs, counts, total

# Interface Gradio com introdução e configuração
with gr.Blocks() as demo:
    # Breve introdução sobre o projeto
    gr.Markdown(
        """
**Projeto desenvolvido para o Programa de Pós-Graduação em Engenharia Biomédica da Universidade de Brasília (PPGEB - UnB), realizado no Campus Gama.**

O mosquito *Aedes aegypti* é um conhecido transmissor de arboviroses que necessita de vigilância constante das entidades responsáveis para seu devido controle populacional e epidemiológico.

Nesse contexto, este projeto surge com o intuito de agilizar o processo de vigilância do controle populacional do mosquito através de uma tentativa de substituir a contagem manual, realizada em laboratório e com extenso uso de tempo e pessoal capacitado, por uma tentativa automática através do uso de redes neurais.

Este projeto conta com uma arquitetura do estado da arte de redes neurais convolucionais chamada **Mask R-CNN**, capaz de realizar a segmentação da imagem com altas métricas de avaliação. Após treinada, a rede é capaz de encontrar os ovos de *Aedes aegypti* na imagem de forma automatizada, realizando inclusive a remoção do fundo automaticamente.

**Para o melhor uso desta ferramenta, recomenda-se:**
- Imagem de papéis-filtro branco.
- Preferencialmente escaneados a 1200 dpi (fundo branco).

**Para utilizá-la**, basta fazer o upload da imagem desejada no campo abaixo e aguardar seu processamento. A imagem será dividida em quatro imagens menores e cada imagem contará com sua contagem individual, bem como a total ao fim do processamento.
        """
    )

    # Título da aplicação
    gr.Markdown("# Detector de Ovos")

    with gr.Row():
        inp    = gr.Image(label="Envie uma imagem", type="pil")
        thresh = gr.Slider(0.0, 1.0, value=0.85, label="Threshold")

    btn = gr.Button("Detectar")

    gallery    = gr.Gallery(label="Segmentações", columns=2, height="auto")
    counts_out = gr.JSON(label="Contagem por pedaço")
    total_out  = gr.Number(label="Total de ovos")

    btn.click(
        fn=run,
        inputs=[inp, thresh],
        outputs=[gallery, counts_out, total_out]
    )

if __name__ == "__main__":
    demo.launch()
