Projeto desenvolvido para o Programa de Pós-Graduação em Engenharia Biomédica da Universidade de Brasília (PPGEB - UnB), realizado no Campus Gama.

O mosquito Aedes aegypti é um conhecido transmissor de arboviroses que necessita de vigilância constante das entidades responsáveis para seu devido controle populacional e epidemiológico.

Nesse contexto, este projeto surge com o intuito de agilizar o processo de vigilância do controle populacional do mosquito através de uma tentativa de substituir a contagem manual, realizada em laboratório e com extenso uso de tempo e pessoal capacitado, por uma tentativa automática através do uso de redes neurais.

Este projeto conta com uma arquitetura do estado da arte de redes neurais convolucionais chamada Mask R-CNN, capaz de realizar a segmentação da imagem com altas métricas de avaliação. Após treinada, a rede é capaz de encontrar os ovos de Aedes aegypti na imagem de forma automatizada, realizando inclusive a remoção do fundo automaticamente.

Para o melhor uso desta ferramenta, recomenda-se:

Imagem de papéis-filtro branco.
Preferencialmente escaneados a 1200 dpi (fundo branco).
Para utilizá-la, basta fazer o upload da imagem desejada no campo abaixo e aguardar seu processamento. A imagem será dividida em quatro imagens menores e cada imagem contará com sua contagem individual, bem como a total ao fim do processamento.

link para o projeto no hugging face: https://huggingface.co/spaces/nativodf/detector_ovos_aedes
