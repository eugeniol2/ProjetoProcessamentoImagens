import inspect
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymupdf
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

import avaliacao
import base_de_dados
import executar_busca_global as busca
import parte1_restauracao as parte1
import parte2_tradicional as parte2
import utilitarios

PASTA_PROJETO = Path(__file__).parent
PASTA_BASE = base_de_dados.pastaDaBase(PASTA_PROJETO)
IMAGEM_PADRAO = "13_img_.png"
INTRODUCAO = "Introdução"
FLUXO_TRADICIONAL = "Fluxo 1 — Tradicional"
FLUXO_MCGLAMERY = "Fluxo 2 — McGlamery"
RESULTADO = "Resultado"
RESULTADOS_GLOBAIS = "Resultados globais"
SLIDES = "Slides da apresentação"
COMBINACAO = "McGlamery + métodos tradicionais"
SECOES = [INTRODUCAO, FLUXO_TRADICIONAL, FLUXO_MCGLAMERY, RESULTADO, RESULTADOS_GLOBAIS]
CHAVE_DO_MODO_SLIDES = "modoSlides"
ARQUIVO_DOS_SLIDES = PASTA_PROJETO / "static" / "slides.pdf"
ENDERECO_DOS_SLIDES = "app/static/slides.pdf"
RESOLUCAO_DOS_SLIDES = 150
GOOGLE_SLIDES = (
    "https://docs.google.com/presentation/d/1rwOhLYTPyqACjoLne-jWar1hVlUJ9XGFF496TNmY9oc"
)
GOOGLE_SLIDES_APRESENTACAO = f"{GOOGLE_SLIDES}/present"
GOOGLE_SLIDES_EMBUTIDO = f"{GOOGLE_SLIDES}/embed?start=false&loop=false&delayms=3000"
ALTURA_DO_GOOGLE_SLIDES = 640
CODIGO_FONTE = "https://github.com/eugeniol2/ProjetoProcessamentoImagens"
CHAVE_DO_SLIDE = "slide"
MODELO = "O modelo de McGlamery-Jaffe"
CANAL_ESCURO = "Etapa 1 — Canal escuro (dark channel)"
COR_DA_AGUA = "Etapa 2 — Cor da água B"
NORMALIZACAO = "Etapa 3a — Normalização"
TRANSMISSAO = "Etapa 3b — Transmissão"
REFINAMENTO = "Etapa 3c — Refinamento"
RECUPERACAO = "Etapa 3d — Recuperação de J"
MUNDO_CINZA = "Etapa 4 — Cor (mundo cinza)"
CLAHE = "Etapa 5 — Contraste (CLAHE)"
HOMOMORFICA = "Etapa 6 — Iluminação (homomórfica)"
ETAPAS_PARTE_1 = [CANAL_ESCURO, COR_DA_AGUA, NORMALIZACAO, TRANSMISSAO, REFINAMENTO, RECUPERACAO]
ETAPAS_PARTE_2 = [MUNDO_CINZA, CLAHE, HOMOMORFICA]
ETAPAS_DO_FLUXO = {
    FLUXO_TRADICIONAL: ETAPAS_PARTE_2,
    FLUXO_MCGLAMERY: [MODELO] + ETAPAS_PARTE_1,
}
PARTE_1 = ":orange[**Restauração McGlamery-Jaffe**] · Parte 1 da proposta, o Fluxo 2"
PARTE_2 = ":blue[**Métodos tradicionais**] · Parte 2 da proposta, o Fluxo 1"
GUIAS_DOS_FLUXOS = {
    FLUXO_TRADICIONAL: "Só os métodos tradicionais, aplicados direto na imagem crua.",
    FLUXO_MCGLAMERY: "Começa pelo modelo de McGlamery-Jaffe e pelas variáveis de que ele precisa. "
    "Etapas 1 a 3d: como cada variável é obtida e como a cena é recuperada. A cena restaurada J "
    "é o resultado do fluxo.",
}
RAIO_MARCA = 20
ESPESSURA_MARCA = 3
VERMELHO = (1.0, 0.0, 0.0)
TAMANHO_DESTAQUE = 5
TAMANHO_RECORTE = 160
FATOR_AMPLIACAO = 2
TAMANHO_AMOSTRA = 120
RAIO_VISTA_FILTRO = 100
COMPRIMENTO_PERFIL = 150
NIVEIS_DE_CINZA = 256
SEM_LIMITE = 0.0
LIMITES_DA_DEMONSTRACAO = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, SEM_LIMITE]
GRADES_DA_DEMONSTRACAO = [1, 2, 4, 8, 16]
BRANCO = (1.0, 1.0, 1.0)
ESPESSURA_GRADE = 2
ESTICAMENTO_EXAGERADO = 10
DESLOCAMENTO_LAB = 128
MUDANCA_MAXIMA_LAB = 60
PASSO_MUDANCA_LAB = 5
TOLERANCIA_DE_MUDANCA = 2
LINHAS_DA_TABELA = 6
PERCENTIS_DO_GRAFICO = [0.5, 99.5]
MARGEM_DO_GRAFICO = 4
TAMANHO_DO_GRAFICO = (7.0, 3.6)
LARGURA_DA_BARRA = 0.85
COR_ABAIXO_DO_TETO = "#3d6f99"
COR_DO_EXCESSO = "#c0392b"
COR_DEVOLVIDA = "#2e8b57"
COR_DO_TETO = "#111111"
COR_DE_DESTAQUE = "#2a78d6"
COR_SEM_DESTAQUE = "#c3c2b7"
COR_DA_SUPERFICIE = "#fcfcfb"
COR_DO_TEXTO = "#0b0b0b"
COR_DO_TEXTO_SECUNDARIO = "#52514e"
COR_DA_GRADE = "#e1e0d9"
LARGURA_DO_GRAFICO_DE_NOTAS = 6.0
ALTURA_DA_BARRA_DE_NOTAS = 0.6
FOLGA_DO_GRAFICO_DO_PSNR = 1.5
LIMITE_DO_GRAFICO_DO_SSIM = 1.4
POSICAO_DO_TITULO = 0.01
PASSO_DAS_MARCAS_DO_PSNR = 5
MARCAS_DO_SSIM = [0, 0.25, 0.5, 0.75, 1.0]
AFASTAMENTO_DO_TEXTO = 0.012
ROTULO_DA_CRUA = "Foto crua"
MUDANCAS_DO_EXEMPLO = [
    ("nada (original)", (0, 0, 0)),
    ("só o L (+40)", (40, 0, 0)),
    ("só o a (+25)", (0, 25, 0)),
    ("só o b (+25)", (0, 0, 25)),
]
NOMES_DOS_CANAIS = ["vermelho", "verde", "azul"]
SIMBOLOS_DO_MODELO = [
    (r"x", "a posição de um pixel na imagem"),
    (r"c", "o canal de cor: R, G ou B"),
    (r"I^c(x)", "a foto captada, que é conhecida"),
    (r"J^c(x)", "a cena sem água, que queremos recuperar"),
    (r"t(x)", "a transmissão: a fração da luz do objeto que chega à câmera, de 0 a 1"),
    (r"B^c", "a cor da água, também chamada de luz de fundo"),
]
HISTORIA_DO_MODELO = {
    "origem": "O modelo de formação da imagem subaquática foi proposto por B. L. McGlamery em "
    "1980, num modelo de computador para câmeras subaquáticas, e ampliado por Jules Jaffe em "
    "1990, que o usou para projetar sistemas de imagem debaixo d'água. Os dois descreveram a "
    "luz que chega à câmera como a soma da luz direta, do espalhamento frontal e do "
    "retroespalhamento. A ideia de uma luz de fundo que se soma à cena é ainda mais antiga: vem "
    "do modelo de visibilidade de Harald Koschmieder, de 1924, criado para explicar por que "
    "objetos distantes somem na névoa.",
    "importancia": "É a base física de quase todos os métodos de restauração subaquática. A "
    "versão simplificada, sem o espalhamento frontal, é a que os métodos de canal escuro usam.",
    "usoNoProjeto": "É o McGlamery do nome do projeto. A Parte 1 inteira existe para estimar as "
    "variáveis dessa fórmula e depois invertê-la.",
}
HISTORIA_DA_AVALIACAO = {
    "origem": "O PSNR é uma medida clássica da engenharia de sinais, usada há décadas para "
    "avaliar a compressão de imagens e vídeos. O SSIM foi proposto por Zhou Wang, Alan Bovik, "
    "Hamid Sheikh e Eero Simoncelli em 2004, para medir a qualidade de um jeito mais próximo da "
    "visão humana, e rendeu aos autores um Emmy de engenharia em 2015.",
    "importancia": "São as duas métricas mais usadas para comparar uma imagem com uma "
    "referência, o que permite comparar o projeto com outros trabalhos da área.",
    "usoNoProjeto": "A proposta define a avaliação com PSNR e SSIM contra as referências do "
    "UIEB, a base criada por Li et al. em 2019.",
}
CODIGO_DAS_ETAPAS = {
    CANAL_ESCURO: (
        [parte1.minimoEntreVerdeEAzul, parte1.erosaoEmTonsDeCinza, parte1.calcularCanalEscuro],
        [],
    ),
    COR_DA_AGUA: (
        [parte1.mascaraDeCandidatos, parte1.posicoesMaisClaras, parte1.estimarCorDaAgua],
        [
            parte1.mascaraTercoSuperior,
            parte1.diferencaEntreCanais,
            parte1.mascaraQuaseBranco,
            parte1.quantidadeMaisClaros,
            parte1.calcularIntensidade,
            parte1.posicaoMaisIntensa,
        ],
    ),
    NORMALIZACAO: ([parte1.normalizarPelaCorDaAgua], [parte1.divisaoSegura]),
    TRANSMISSAO: ([parte1.calcularTransmissao], [parte1.calcularCanalEscuro]),
    REFINAMENTO: (
        [parte1.filtroGuiado, parte1.refinarTransmissao],
        [parte1.calcularIntensidade],
    ),
    RECUPERACAO: (
        [parte1.limitarTransmissao, parte1.recuperarCena],
        [parte1.adicionarEixoDeCanal],
    ),
    MUNDO_CINZA: (
        [parte2.ganhosDoMundoCinza, parte2.aplicarMundoCinza],
        [parte2.mediasDosCanais, parte2.limitarGanhos],
    ),
    CLAHE: (
        [parte2.equalizarLuminosidade, parte2.aplicarClahe],
        [parte2.rgbParaLab, parte2.labParaRgb, parte2.trocarLuminosidade],
    ),
    HOMOMORFICA: (
        [
            parte2.filtroEnfaseAltasFrequencias,
            parte2.filtrarLuminosidade,
            parte2.aplicarFiltragemHomomorfica,
        ],
        [
            parte2.paraDominioLogaritmico,
            parte2.espectroCentralizado,
            parte2.distanciaAoCentroAoQuadrado,
            parte2.voltarAoDominioEspacial,
            parte2.sairDoDominioLogaritmico,
            parte2.paraNiveisDeLuminosidade,
        ],
    ),
}
ROTULOS_RESTAURACAO = ["1. Canal escuro", "2. Cor da água", "3. Transmissão e recuperação"]
ROTULOS_TRADICIONAIS = ["4. Mundo cinza", "5. CLAHE", "6. Homomórfica"]
ESTILO_DIAGRAMA = (
    "display:grid;grid-template-columns:110px 28px 1fr 28px 120px;"
    "gap:8px 4px;align-items:center;background:#f7f9fb;border:1px solid #c9d3dd;"
    "border-radius:14px;padding:14px 16px;font-family:sans-serif;color:#1f2937"
)
ESTILO_TITULO_FLUXO = "grid-column:1 / -1;font-weight:700;font-size:15px;margin-top:6px"
ESTILO_NO = (
    "background:#ffffff;border:1px solid #6b7280;border-radius:10px;padding:10px 6px;"
    "text-align:center;font-size:14px"
)
ESTILO_SETA = "text-align:center;color:#3d6f99;font-size:20px"
ESTILO_TITULO_GRUPO = "text-align:center;font-size:13px;font-weight:600;margin-bottom:6px"
ESTILO_ETIQUETAS = "display:flex;align-items:center;gap:6px"
ESTILO_SETA_ETIQUETA = "color:#3d6f99"
GRUPO_TRADICIONAL = (
    "background:#eaf2fa;border:1px solid #3d6f99;border-radius:12px;padding:8px 10px 10px;"
    "color:#24527a",
    "flex:1;background:#d3e5f5;border:1px solid #3d6f99;border-radius:8px;padding:8px 4px;"
    "text-align:center;font-size:13px;color:#1f2937",
)
GRUPO_MCGLAMERY = (
    "background:#fdf1e0;border:1px solid #c27c0e;border-radius:12px;padding:8px 10px 10px;"
    "color:#8a5300",
    "flex:1;background:#f9d9a6;border:1px solid #c27c0e;border-radius:8px;padding:8px 4px;"
    "text-align:center;font-size:13px;color:#1f2937",
)
PARAMETROS_DO_PROJETO = {
    "janela": parte1.JANELA_CANAL_ESCURO,
    "raio": parte1.RAIO_FILTRO_GUIADO,
    "piso": parte1.TRANSMISSAO_MINIMA,
    "ganhoMaximo": parte2.GANHO_MAXIMO,
    "limiteClahe": parte2.LIMITE_CONTRASTE_CLAHE,
    "gradeClahe": parte2.BLOCOS_CLAHE,
    "ganhoBaixas": parte2.GANHO_BAIXAS_FREQUENCIAS,
    "ganhoAltas": parte2.GANHO_ALTAS_FREQUENCIAS,
    "corte": parte2.FREQUENCIA_DE_CORTE,
}
NOMES_DOS_PARAMETROS = {
    "janela": "Janela do canal escuro",
    "raio": "Raio do filtro guiado",
    "piso": "Piso da transmissão t0",
    "ganhoMaximo": "Limite do ganho do mundo cinza",
    "limiteClahe": "Limite de contraste do CLAHE",
    "gradeClahe": "Grade do CLAHE",
    "ganhoBaixas": "Ganho da iluminação (homomórfica)",
    "ganhoAltas": "Ganho dos detalhes (homomórfica)",
    "corte": "Frequência de corte (homomórfica)",
}
JANELAS_DA_DEMONSTRACAO = list(range(3, 32, 2))
ESCALA_DO_MAPA = 4
LIMITACAO_DO_PROJETO = (
    "**Limitação.** Os parâmetros foram escolhidos na imagem piloto, a 13_img_, e não consegui "
    "definir um conjunto de valores que funcione bem em todas as imagens. Nas 20 fotos da "
    "amostra do projeto, com os mesmos valores, o Fluxo 2 ficou à frente do Fluxo 1 em só 5 "
    "delas no PSNR, e 4 no SSIM. Na média, ficou abaixo até da foto crua: 15,31 dB, contra "
    "15,78 dB da crua e 16,96 dB do Fluxo 1. A imagem piloto é um caso favorável à restauração. "
    "Por isso os parâmetros ficam ajustáveis em cada etapa: dá para escolher outra foto na barra "
    "lateral e testar se ela fica melhor ou pior. A seção Resultados globais procura os valores "
    "que funcionam melhor na base inteira."
)
ARQUIVOS_DA_BUSCA = ["fluxo1", "fluxo2", "combinacao", "por_imagem", "melhores_por_foto"]
SAIDAS_DA_BUSCA = {FLUXO_TRADICIONAL: "fluxo1", FLUXO_MCGLAMERY: "fluxo2", COMBINACAO: "combinacao"}
FOLGA_DO_GRAFICO_DO_EFEITO = 1.9
VALORES_PARA_TER_MEIO = 3
LARGURA_DO_GRAFICO_DO_EFEITO = 6.0
ALTURA_POR_BARRA = 0.42
ALTURA_EXTRA_DO_GRAFICO = 0.9
LARGURA_POR_VALOR = 0.75
MARGEM_DO_GRAFICO_DE_CONTAGEM = 0.6
ALTURA_DO_GRAFICO_DE_CONTAGEM = 3.0
LARGURA_DA_COLUNA = 0.7
FOLGA_DAS_COLUNAS = 1.18
LARGURA_DO_TITULO_DO_PAINEL = 18
CORES_DOS_METODOS = ["#2a78d6", "#eb6834", "#1baf7a"]
OPACIDADE_SEM_ESCOLHA = 0.4
METODOS_DOS_PARAMETROS = {
    "ganhoMaximo": "Mundo cinza",
    "limiteClahe": "CLAHE",
    "gradeClahe": "CLAHE",
    "ganhoBaixas": "Homomórfica",
    "ganhoAltas": "Homomórfica",
    "janela": "Canal escuro",
    "raio": "Filtro guiado",
    "piso": "Recuperação de J",
}
COR_DO_DIVE = "#2a78d6"
COR_DO_PROJETO = "#e34948"
ALTURA_DA_COMPARACAO = 2.0
ALTURA_DA_BARRA_DE_FORA = 0.7
ALTURA_DA_BARRA_DE_DENTRO = 0.36
LIMITES_DA_BARRA_UNICA = (-0.6, 0.6)
FOLGA_DA_COMPARACAO_DO_PSNR = 1.3
LIMITE_DA_COMPARACAO_DO_SSIM = 1.25
ESPESSURA_DO_CONTORNO = 1.5
COMO_O_UIEB_AVALIA = (
    "- **As referências.** Cada foto foi melhorada por 12 métodos, e 50 voluntários escolheram a "
    "melhor versão, comparando duas a duas.\n"
    "- **As fotos sem referência.** As 60 fotos em que nenhuma versão agradou ficaram sem "
    "referência e não entram nesta avaliação.\n"
    "- **A nota.** Cada método é aplicado às 890 fotos, e a nota é a média do PSNR e do SSIM "
    "contra as referências.\n"
    "- **Notas sem referência.** Os autores testaram notas que olham só a saída, a UCIQE e a "
    "UIQM, e mostraram que elas nem sempre concordam com o olho humano."
)
NOTAS_DO_ARTIGO_DO_UIEB = {
    "dive+": (20.84, 0.871),
    "Fusão": (18.75, 0.816),
    "Histogram prior": (16.01, 0.589),
    "Red Channel": (14.89, 0.597),
    "GDCP": (12.53, 0.550),
    "UDCP": (11.03, 0.500),
}


def executarFluxo(entrada, parametros):
    cor = parte2.aplicarMundoCinza(entrada, parametros["ganhoMaximo"])
    contraste = parte2.aplicarClahe(cor, parametros["limiteClahe"], parametros["gradeClahe"])
    final = parte2.aplicarFiltragemHomomorfica(
        contraste, parametros["ganhoBaixas"], parametros["ganhoAltas"], parametros["corte"]
    )

    return {"entrada": entrada, "cor": cor, "contraste": contraste, "final": final}


@st.cache_resource(max_entries=2, show_spinner="Processando a imagem...")
def processarImagem(nomeImagem, parametros):
    crua, referencia = base_de_dados.carregarPar(nomeImagem, PASTA_BASE)
    canalEscuro = parte1.calcularCanalEscuro(crua, parametros["janela"])
    corDaAgua, posicaoDaAgua = parte1.estimarCorDaAgua(crua, canalEscuro)
    normalizada = parte1.normalizarPelaCorDaAgua(crua, corDaAgua)
    transmissao = parte1.calcularTransmissao(normalizada, parametros["janela"])
    transmissaoRefinada = parte1.refinarTransmissao(transmissao, crua, parametros["raio"])
    restaurada = parte1.recuperarCena(crua, corDaAgua, transmissaoRefinada, parametros["piso"])

    resultados = {
        "nomeImagem": nomeImagem,
        "parametros": parametros,
        "crua": crua,
        "referencia": referencia,
        "canalEscuro": canalEscuro,
        "corDaAgua": corDaAgua,
        "posicaoDaAgua": posicaoDaAgua,
        "normalizada": normalizada,
        "transmissao": transmissao,
        "transmissaoRefinada": transmissaoRefinada,
        "restaurada": restaurada,
        FLUXO_TRADICIONAL: executarFluxo(crua, parametros),
        FLUXO_MCGLAMERY: {"entrada": crua, "final": restaurada},
        COMBINACAO: executarFluxo(restaurada, parametros),
    }
    resultados["notas"] = calcularNotas(resultados)
    resultados["notasEntreFluxos"] = notasEntreFluxos(resultados)

    return resultados


def numeroBrasileiro(valor):
    return f"{valor:g}".replace(".", ",")


def caixaHtml(conteudo, estilo):
    return f'<div style="{estilo}">{conteudo}</div>'


def linhaDeEtiquetas(rotulos, estiloEtiqueta):
    seta = f'<span style="{ESTILO_SETA_ETIQUETA}">→</span>'
    etiquetas = seta.join(caixaHtml(rotulo, estiloEtiqueta) for rotulo in rotulos)

    return caixaHtml(etiquetas, ESTILO_ETIQUETAS)


def caixaDeGrupo(titulo, rotulos, estilosDoGrupo):
    estiloGrupo, estiloEtiqueta = estilosDoGrupo
    conteudo = caixaHtml(titulo, ESTILO_TITULO_GRUPO) + linhaDeEtiquetas(rotulos, estiloEtiqueta)

    return caixaHtml(conteudo, estiloGrupo)


def linhaDoFluxo(titulo, blocoDoFluxo):
    seta = caixaHtml("→", ESTILO_SETA)

    return "".join(
        [
            caixaHtml(titulo, ESTILO_TITULO_FLUXO),
            caixaHtml("Imagem crua", ESTILO_NO),
            seta,
            blocoDoFluxo,
            seta,
            caixaHtml("Resultado<br>PSNR e SSIM", ESTILO_NO),
        ]
    )


def diagramaDosFluxos():
    tradicionais = caixaDeGrupo("Métodos tradicionais", ROTULOS_TRADICIONAIS, GRUPO_TRADICIONAL)
    restauracao = caixaDeGrupo("Restauração McGlamery-Jaffe", ROTULOS_RESTAURACAO, GRUPO_MCGLAMERY)
    linhaTradicional = linhaDoFluxo(FLUXO_TRADICIONAL, tradicionais)
    linhaMcglamery = linhaDoFluxo(FLUXO_MCGLAMERY, restauracao)

    return caixaHtml(linhaTradicional + linhaMcglamery, ESTILO_DIAGRAMA)


def descreverImagem(nomeImagem):
    if nomeImagem == IMAGEM_PADRAO:
        return "Imagem piloto do projeto: todos os parâmetros foram escolhidos nela."

    return (
        "Outra foto da base. Os parâmetros do projeto foram escolhidos na imagem piloto, a "
        "13_img_."
    )


def cinzaParaRgb(canal):
    return np.stack([canal, canal, canal], axis=-1)


def copiaEmRgb(imagem):
    return cinzaParaRgb(imagem) if imagem.ndim == 2 else imagem.copy()


def marcarPosicao(imagem, posicao):
    marcada = copiaEmRgb(imagem)
    linha, coluna = posicao
    cv2.circle(marcada, (int(coluna), int(linha)), RAIO_MARCA, VERMELHO, ESPESSURA_MARCA)

    return marcada


def destacarPosicoes(canal, linhas, colunas):
    mascara = np.zeros(canal.shape, dtype=np.uint8)
    mascara[linhas, colunas] = 1
    elemento = np.ones((TAMANHO_DESTAQUE, TAMANHO_DESTAQUE), dtype=np.uint8)
    destacada = cinzaParaRgb(canal)
    destacada[cv2.dilate(mascara, elemento).astype(bool)] = VERMELHO

    return destacada


def amostraDeCor(cor):
    return np.ones((TAMANHO_AMOSTRA, TAMANHO_AMOSTRA, 3)) * cor


def centroDaImagem(imagem):
    return imagem.shape[0] // 2, imagem.shape[1] // 2


def regiaoDaAgua(imagem):
    return imagem.shape[0] // 6, imagem.shape[1] // 6


def recorteAmpliado(imagem, centro):
    tamanho = min(TAMANHO_RECORTE, imagem.shape[0], imagem.shape[1])
    topo = int(np.clip(centro[0] - tamanho // 2, 0, imagem.shape[0] - tamanho))
    esquerda = int(np.clip(centro[1] - tamanho // 2, 0, imagem.shape[1] - tamanho))
    recorte = np.ascontiguousarray(imagem[topo : topo + tamanho, esquerda : esquerda + tamanho])

    return cv2.resize(
        recorte, None, fx=FATOR_AMPLIACAO, fy=FATOR_AMPLIACAO, interpolation=cv2.INTER_NEAREST
    )


def normalizarParaExibicao(valores):
    return (valores - valores.min()) / (valores.max() - valores.min())


def luminosidadeDe(imagem):
    return parte2.rgbParaLab(imagem)[:, :, parte2.LUMINOSIDADE]


def histogramaDaLuminosidade(imagem):
    return np.bincount(luminosidadeDe(imagem).ravel(), minlength=NIVEIS_DE_CINZA)


def equalizacaoGlobal(imagem):
    lab = parte2.rgbParaLab(imagem)
    equalizada = cv2.equalizeHist(np.ascontiguousarray(lab[:, :, parte2.LUMINOSIDADE]))

    return parte2.labParaRgb(parte2.trocarLuminosidade(lab, equalizada))


def bordaQueFalta(tamanho, blocos):
    return blocos - tamanho % blocos


def luminosidadeComBorda(luminosidade, blocos):
    altura, largura = luminosidade.shape
    if altura % blocos == 0 and largura % blocos == 0:
        return luminosidade

    return cv2.copyMakeBorder(
        luminosidade,
        0,
        bordaQueFalta(altura, blocos),
        0,
        bordaQueFalta(largura, blocos),
        cv2.BORDER_REFLECT_101,
    )


def tamanhoDoBloco(luminosidade, blocos):
    altura, largura = luminosidadeComBorda(luminosidade, blocos).shape

    return altura // blocos, largura // blocos


def blocoDaPosicao(alturaBloco, larguraBloco, posicao):
    return posicao[0] // alturaBloco, posicao[1] // larguraBloco


def recorteDoBloco(canal, alturaBloco, larguraBloco, bloco):
    linha, coluna = bloco

    return canal[
        linha * alturaBloco : (linha + 1) * alturaBloco,
        coluna * larguraBloco : (coluna + 1) * larguraBloco,
    ]


def tetoDoBloco(pixelsNoBloco, limite):
    if limite == SEM_LIMITE:
        return pixelsNoBloco

    return max(int(limite * pixelsNoBloco / NIVEIS_DE_CINZA), 1)


def excessoAcimaDoTeto(histograma, teto):
    return int(np.maximum(histograma - teto, 0).sum())


def redistribuirExcesso(cortado, excesso):
    porNivel, resto = divmod(excesso, NIVEIS_DE_CINZA)
    redistribuido = cortado + porNivel
    if resto:
        redistribuido[:: max(NIVEIS_DE_CINZA // resto, 1)][:resto] += 1

    return redistribuido


def cortarHistograma(histograma, teto):
    return redistribuirExcesso(np.minimum(histograma, teto), excessoAcimaDoTeto(histograma, teto))


def regraDeTroca(histograma, pixelsNoBloco):
    acumulado = np.cumsum(histograma) * (NIVEIS_DE_CINZA - 1) / pixelsNoBloco

    return np.clip(np.round(acumulado), 0, NIVEIS_DE_CINZA - 1)


def esticamentoMaximo(histograma, pixelsNoBloco):
    return histograma.max() * (NIVEIS_DE_CINZA - 1) / pixelsNoBloco


def analisarBloco(luminosidade, limite, blocos):
    alturaBloco, larguraBloco = tamanhoDoBloco(luminosidade, blocos)
    bloco = blocoDaPosicao(alturaBloco, larguraBloco, regiaoDaAgua(luminosidade))
    pixels = recorteDoBloco(
        luminosidadeComBorda(luminosidade, blocos), alturaBloco, larguraBloco, bloco
    )
    histograma = np.bincount(pixels.ravel(), minlength=NIVEIS_DE_CINZA)
    teto = tetoDoBloco(pixels.size, limite)

    return {
        "alturaBloco": alturaBloco,
        "larguraBloco": larguraBloco,
        "bloco": bloco,
        "pixels": pixels,
        "histograma": histograma,
        "teto": teto,
        "excesso": excessoAcimaDoTeto(histograma, teto),
        "cortado": cortarHistograma(histograma, teto),
    }


def desenharGrade(imagem, alturaBloco, larguraBloco, bloco):
    marcada = copiaEmRgb(imagem)
    altura, largura = marcada.shape[:2]
    for linha in range(alturaBloco, altura, alturaBloco):
        cv2.line(marcada, (0, linha), (largura - 1, linha), BRANCO, ESPESSURA_GRADE)
    for coluna in range(larguraBloco, largura, larguraBloco):
        cv2.line(marcada, (coluna, 0), (coluna, altura - 1), BRANCO, ESPESSURA_GRADE)
    linha, coluna = bloco
    inicio = (coluna * larguraBloco, linha * alturaBloco)
    fim = ((coluna + 1) * larguraBloco, (linha + 1) * alturaBloco)
    cv2.rectangle(marcada, inicio, fim, VERMELHO, ESPESSURA_MARCA)

    return marcada


def espectroLogaritmico(imagem):
    espectro = parte2.espectroCentralizado(parte2.paraDominioLogaritmico(luminosidadeDe(imagem)))

    return normalizarParaExibicao(np.log1p(np.abs(espectro)))


def ganhosDoFiltro(altura, largura, parametros):
    return parte2.filtroEnfaseAltasFrequencias(
        altura, largura, parametros["ganhoBaixas"], parametros["ganhoAltas"], parametros["corte"]
    )


def vistaDoFiltro(altura, largura, parametros):
    ganhos = ganhosDoFiltro(altura, largura, parametros)
    linha, coluna = altura // 2, largura // 2
    raio = min(RAIO_VISTA_FILTRO, linha, coluna)
    janela = ganhos[linha - raio : linha + raio, coluna - raio : coluna + raio]
    amplitude = parametros["ganhoAltas"] - parametros["ganhoBaixas"]
    if amplitude <= 0:
        return np.zeros_like(janela)

    return (janela - parametros["ganhoBaixas"]) / amplitude


def perfilDoFiltro(altura, largura, parametros):
    ganhos = ganhosDoFiltro(altura, largura, parametros)
    linha, coluna = altura // 2, largura // 2

    return ganhos[linha, coluna : coluna + COMPRIMENTO_PERFIL]


def tabelaDeGanhos(imagem, ganhoMaximo):
    medias = parte2.mediasDosCanais(imagem)

    return {
        "canal": ["R", "G", "B"],
        "média do canal": medias.round(3),
        "ganho sem limite": (medias.mean() / medias).round(2),
        "ganho usado": parte2.ganhosDoMundoCinza(imagem, ganhoMaximo).round(2),
    }


def notasDaImagem(imagem, referencia):
    return avaliacao.calcularPsnr(imagem, referencia), avaliacao.calcularSsim(imagem, referencia)


def calcularNotas(resultados):
    referencia = resultados["referencia"]
    imagens = {
        "Imagem crua": resultados["crua"],
        FLUXO_TRADICIONAL: resultados[FLUXO_TRADICIONAL]["final"],
        FLUXO_MCGLAMERY: resultados[FLUXO_MCGLAMERY]["final"],
        COMBINACAO: resultados[COMBINACAO]["final"],
    }

    return {rotulo: notasDaImagem(imagem, referencia) for rotulo, imagem in imagens.items()}


def notasEntreFluxos(resultados):
    return notasDaImagem(
        resultados[FLUXO_MCGLAMERY]["final"], resultados[FLUXO_TRADICIONAL]["final"]
    )


def diferencaEntreFluxos(resultados):
    tradicional = resultados[FLUXO_TRADICIONAL]["final"]
    mcglamery = resultados[FLUXO_MCGLAMERY]["final"]

    return np.abs(mcglamery - tradicional).mean(axis=2) * utilitarios.INTENSIDADE_MAXIMA


def mapaDaDiferenca(diferenca):
    niveis = np.clip(diferenca * ESCALA_DO_MAPA, 0, utilitarios.INTENSIDADE_MAXIMA)

    return cv2.applyColorMap(niveis.astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]


def textoDoPsnr(psnr):
    if np.isinf(psnr):
        return "infinito, porque as saídas são iguais"

    return f"{decimalBrasileiro(psnr, 2)} dB"


def mostrarDiferencaEntreFluxos(resultados):
    psnr, ssim = resultados["notasEntreFluxos"]
    diferenca = diferencaEntreFluxos(resultados)
    altura = diferenca.shape[0]
    st.markdown("**Quanto os dois fluxos diferem entre si**")
    esquerda, direita = st.columns(2)
    esquerda.image(
        mapaDaDiferenca(diferenca),
        caption="Onde as saídas dos dois fluxos diferem: quanto mais claro, maior a diferença",
    )
    with direita:
        st.write(
            f"O Fluxo 2 comparado diretamente com o Fluxo 1: PSNR {textoDoPsnr(psnr)} e SSIM "
            f"{decimalBrasileiro(ssim, 3)}. Quanto mais altos, mais parecidas são as duas saídas."
        )
        st.write(
            "Diferença média, de 0 a 255: "
            f"{decimalBrasileiro(diferenca[: altura // 3].mean(), 1)} no terço de cima e "
            f"{decimalBrasileiro(diferenca[2 * altura // 3 :].mean(), 1)} no terço de baixo."
        )
        st.info(
            "Essa comparação mostra quanto e onde as saídas dos dois fluxos diferem, mas não diz "
            "qual é melhor. Para isso servem as notas contra a referência, acima."
        )


def rotuloDecimalDoEixo(valor, posicao):
    return numeroBrasileiro(round(valor, 2))


def rotulosDoGrafico(notas):
    return [ROTULO_DA_CRUA if rotulo == "Imagem crua" else rotulo for rotulo in notas]


def barrasEmDestaque(valores, casas):
    arredondados = [round(valor, casas) for valor in valores]

    return [valor == max(arredondados) for valor in arredondados]


def textoDaBarra(valor, casas, destaque):
    texto = decimalBrasileiro(valor, casas)

    return f"{texto} · melhor" if destaque else texto


def escreverValoresNasBarras(eixo, valores, casas, destaques, limite):
    afastamento = limite * AFASTAMENTO_DO_TEXTO
    for posicao, (valor, destaque) in enumerate(zip(valores, destaques)):
        eixo.text(
            valor + afastamento,
            posicao,
            textoDaBarra(valor, casas, destaque),
            va="center",
            fontsize=9,
            color=COR_DO_TEXTO,
            fontweight="bold" if destaque else "normal",
        )


def formatarEixoDasNotas(eixo, rotulos, limite, marcas):
    eixo.set_facecolor(COR_DA_SUPERFICIE)
    eixo.set_yticks(range(len(rotulos)), rotulos, fontsize=9, color=COR_DO_TEXTO)
    eixo.invert_yaxis()
    eixo.set_xlim(0, limite)
    eixo.set_xticks(marcas)
    eixo.xaxis.set_major_formatter(FuncFormatter(rotuloDecimalDoEixo))
    eixo.grid(axis="x", color=COR_DA_GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    eixo.spines[["top", "right", "left"]].set_visible(False)
    eixo.spines["bottom"].set_color(COR_SEM_DESTAQUE)
    eixo.tick_params(axis="x", labelsize=8, colors=COR_DO_TEXTO_SECUNDARIO)
    eixo.tick_params(axis="y", length=0)


def graficoDeNotas(notas, indice, casas, limite, marcas, titulo):
    valores = [nota[indice] for nota in notas.values()]
    destaques = barrasEmDestaque(valores, casas)
    cores = [COR_DE_DESTAQUE if destaque else COR_SEM_DESTAQUE for destaque in destaques]
    altura = ALTURA_POR_BARRA * len(valores) + ALTURA_EXTRA_DO_GRAFICO
    figura = Figure(
        figsize=(LARGURA_DO_GRAFICO_DE_NOTAS, altura),
        layout="constrained",
        facecolor=COR_DA_SUPERFICIE,
    )
    eixo = figura.subplots()
    eixo.barh(range(len(valores)), valores, height=ALTURA_DA_BARRA_DE_NOTAS, color=cores)
    escreverValoresNasBarras(eixo, valores, casas, destaques, limite)
    formatarEixoDasNotas(eixo, rotulosDoGrafico(notas), limite, marcas)
    figura.suptitle(titulo, x=POSICAO_DO_TITULO, ha="left", fontsize=10, color=COR_DO_TEXTO)

    return figura


def graficoDoPsnr(notas):
    limite = max(psnr for psnr, ssim in notas.values()) * FOLGA_DO_GRAFICO_DO_PSNR
    marcas = np.arange(0, limite, PASSO_DAS_MARCAS_DO_PSNR)

    return graficoDeNotas(
        notas, 0, 2, limite, marcas, "PSNR (dB): quanto maior, mais perto da referência"
    )


def graficoDoSsim(notas):
    return graficoDeNotas(
        notas,
        1,
        3,
        LIMITE_DO_GRAFICO_DO_SSIM,
        MARCAS_DO_SSIM,
        "SSIM, de 0 a 1: quanto maior, mais parecida a estrutura",
    )


def mostrarGraficosDasNotas(notas):
    st.markdown("**Qual saída chegou mais perto da referência**")
    esquerda, direita = st.columns(2)
    esquerda.pyplot(graficoDoPsnr(notas))
    direita.pyplot(graficoDoSsim(notas))
    st.caption(
        "Em azul e com a palavra 'melhor', a maior nota de cada gráfico. Notas iguais até a "
        "última casa mostrada contam como empate e aparecem juntas."
    )


def tabelaDeNotas(notas):
    return {
        "imagem": list(notas),
        "PSNR (dB)": [round(psnr, 2) for psnr, ssim in notas.values()],
        "SSIM": [round(ssim, 3) for psnr, ssim in notas.values()],
    }


def direcao(diferenca):
    return "acima" if diferenca >= 0 else "abaixo"


def trechoDaDiferenca(diferenca, casas, unidade, metrica):
    valor = round(abs(diferenca), casas)
    if valor == 0:
        return f"com a mesma nota do Fluxo 1 no {metrica}"

    return f"{numeroBrasileiro(valor)}{unidade} {direcao(diferenca)} do Fluxo 1 no {metrica}"


def comparacaoComTradicional(notas, rotulo):
    psnrTradicional, ssimTradicional = notas[FLUXO_TRADICIONAL]
    psnr, ssim = notas[rotulo]
    trechoPsnr = trechoDaDiferenca(psnr - psnrTradicional, 2, " dB", "PSNR")
    trechoSsim = trechoDaDiferenca(ssim - ssimTradicional, 3, "", "SSIM")

    return f"{trechoPsnr} e {trechoSsim}"


def descreverComparacao(notas):
    return (
        f"Nesta imagem, o Fluxo 2 ficou {comparacaoComTradicional(notas, FLUXO_MCGLAMERY)}. "
        f"{COMBINACAO} ficou {comparacaoComTradicional(notas, COMBINACAO)}."
    )


def textosDasEtapas():
    janela = parte1.JANELA_CANAL_ESCURO
    fracao = numeroBrasileiro(parte1.FRACAO_MAIS_CLAROS * 100)
    limite = parte1.TRANSMISSAO_MINIMA
    blocos = parte2.BLOCOS_CLAHE

    return {
        CANAL_ESCURO: {
            "problema": "A água cobre a cena com um véu que fica mais forte quanto mais longe "
            "está o objeto. Para tirar esse véu, é preciso saber quanto véu existe em cada ponto, "
            "e a foto não traz essa informação.",
            "motivo": "O canal escuro estima o véu a partir da própria foto. Em objetos quase "
            "sempre existe algum pixel escuro em algum canal de cor, e o véu da água clareia "
            "tudo. Então, onde o canal mais escuro de uma região está claro, há muita água. O "
            "vermelho fica de fora porque a água o absorve e ele ficaria escuro na imagem "
            "inteira. Essa é a versão subaquática do método, o UDCP.",
            "comoFunciona": "Em cada pixel, fica com o menor valor entre o verde e o azul. Depois, "
            f"uma erosão {janela}×{janela} do OpenCV troca cada pixel pelo menor valor da "
            "vizinhança.",
            "observar": "claro onde só há água e escuro nos objetos.",
        },
        COR_DA_AGUA: {
            "problema": "A equação do modelo precisa da cor da água, e essa cor muda de uma foto "
            "para outra.",
            "motivo": "Os pixels com mais véu, que são os mais claros do canal escuro, são os que "
            "mais se parecem com água pura, então a cor deles é a cor da água. A busca fica no "
            "terço superior, onde a água costuma estar, e ignora pixels quase brancos. Sem essas "
            "restrições, no primeiro teste a cor da água foi tirada do cilindro branco do "
            "mergulhador.",
            "comoFunciona": "A área de busca é o terço superior da foto, sem os pixels quase "
            f"brancos. Dentro dela, pega os {fracao}% pixels mais claros do canal escuro e, entre "
            "eles, o mais claro na foto original.",
            "observar": "o círculo vermelho deve cair na água, longe de objetos claros.",
        },
        NORMALIZACAO: {
            "problema": "O véu precisa ser medido em relação à cor da água daquela foto, que é "
            "diferente em cada imagem.",
            "motivo": "Dividir pela cor da água faz a água valer 1 nos três canais. Assim, o "
            "canal escuro da imagem dividida passa a medir diretamente quanto de cada pixel é "
            "água, numa escala de 0 a 1, qualquer que seja a cor da água.",
            "comoFunciona": "Cada canal da imagem é dividido pelo canal correspondente da cor da "
            "água B.",
            "observar": "a água fica quase branca e os objetos ficam mais escuros que ela.",
        },
        TRANSMISSAO: {
            "problema": "Para corrigir cada pixel na medida certa, é preciso saber quanto da luz "
            "do objeto chegou à câmera e quanto foi encoberto pela água.",
            "motivo": "A transmissão sai direto da imagem normalizada, sem precisar de sensor de "
            "distância. Ela vale perto de 1 nos objetos próximos e perto de 0 onde só se vê "
            "água, então funciona como um mapa de profundidade.",
            "comoFunciona": "É 1 menos o canal escuro da imagem normalizada.",
            "observar": "objetos próximos claros e a água do fundo escura, ainda com os blocos "
            "da erosão.",
        },
        REFINAMENTO: {
            "problema": "A erosão deixa a transmissão em blocos quadrados. Usada assim, ela "
            "criaria manchas e halos em volta dos objetos na imagem restaurada.",
            "motivo": "O filtro guiado suaviza a transmissão seguindo as bordas da imagem "
            "original. Ele remove os blocos e mantém o contorno dos objetos, o que um desfoque "
            "comum não faria.",
            "comoFunciona": "Usa a foto original em tons de cinza como guia, em janelas de raio "
            f"{parte1.RAIO_FILTRO_GUIADO}. Onde a foto tem borda, a transmissão passa a seguir "
            "essa borda. Onde a foto é lisa, a transmissão vira a média da região, e os blocos "
            "somem. No código, é uma chamada pronta do módulo extra do OpenCV (ximgproc).",
            "observar": "os blocos somem e o contorno dos objetos continua nítido.",
        },
        RECUPERACAO: {
            "problema": "Cada pixel da foto é uma mistura da cor do objeto com a cor da água. É "
            "preciso separar os dois para recuperar a cena.",
            "motivo": "Com a cor da água e a transmissão estimadas, a equação de McGlamery-Jaffe "
            "pode ser invertida para tirar o véu. O piso t0 evita dividir por transmissões muito "
            "pequenas, que multiplicariam o ruído.",
            "comoFunciona": "Faz a conta do modelo de McGlamery ao contrário, com um piso t0 = "
            f"{numeroBrasileiro(limite)} na transmissão. Com esse piso, a diferença entre cada "
            "pixel e a cor da água é ampliada no máximo "
            f"{numeroBrasileiro(round(1 / limite, 1))} vezes.",
            "observar": "mais contraste e menos véu. A cor continua azul-esverdeada, porque a "
            "restauração não corrige a cor. Esta cena é o resultado final do Fluxo 2.",
        },
        MUNDO_CINZA: {
            "problema": "A água deixa a imagem azul-esverdeada, porque absorve o vermelho.",
            "motivo": "O mundo cinza (gray world) é a correção de cor automática mais simples: "
            "supõe que a média da cena é cinza e multiplica cada canal por um ganho que iguala "
            "as médias, usando só a própria imagem. O ganho é limitado a "
            f"{numeroBrasileiro(parte2.GANHO_MAXIMO)} porque o vermelho quase não existe; sem o "
            "limite, ele seria multiplicado demais e a cena ficaria rosa.",
            "comoFunciona": "Mede a média de cada canal de cor e multiplica cada canal por um "
            "ganho que deixa as três médias iguais. O vermelho, o mais fraco, recebe o maior "
            "ganho, e o teto segura esse ganho.",
            "observar": "menos verde, sem a cena ficar rosa.",
        },
        CLAHE: {
            "problema": "Os tons da imagem ficam espremidos numa faixa estreita, então o "
            "contraste é baixo e os detalhes se perdem.",
            "motivo": "A equalização de histograma espalha os tons e aumenta o contraste. O CLAHE "
            "faz isso por regiões e com um teto, para não amplificar o ruído nem os blocos da "
            "compressão JPEG na água, o que acontece com a equalização global. É aplicado só no "
            "canal de brilho, o L do LAB, para não mudar as cores.",
            "comoFunciona": f"Divide a imagem numa grade de {blocos}×{blocos}, ou seja, "
            f"{blocos * blocos} blocos, e equaliza cada bloco separado. O limite de contraste, "
            f"{numeroBrasileiro(parte2.LIMITE_CONTRASTE_CLAHE)} no projeto, funciona como um "
            "teto: nenhum tom pode ganhar destaque demais, e assim o ruído da água não é "
            "esticado. Os blocos vizinhos são misturados, para não aparecerem emendas. No código, "
            "tudo isso é uma chamada pronta do OpenCV.",
            "observar": "mais contraste, com as mesmas cores.",
        },
        HOMOMORFICA: {
            "problema": "A luz chega de forma desigual na cena. Algumas partes ficam claras "
            "demais e outras escuras demais, escondendo detalhes.",
            "motivo": "A filtragem homomórfica separa a iluminação, que varia devagar, dos "
            "detalhes, que variam rápido. No domínio da frequência, ela atenua a iluminação e "
            "reforça os detalhes, sem mudar o brilho médio.",
            "comoFunciona": "Passa o brilho para o logaritmo, onde iluminação e detalhes se "
            "separam, e depois para a frequência com a FFT do NumPy. As baixas frequências, que "
            "guardam a iluminação, são multiplicadas por "
            f"{numeroBrasileiro(parte2.GANHO_BAIXAS_FREQUENCIAS)}, e as altas, que guardam os "
            f"detalhes, por {numeroBrasileiro(parte2.GANHO_ALTAS_FREQUENCIAS)}. O brilho médio "
            "fica igual, e a imagem volta ao normal pelo caminho inverso.",
            "observar": "luz mais uniforme e detalhes mais marcados.",
        },
    }


def historiaDasEtapas():
    limite = numeroBrasileiro(parte1.TRANSMISSAO_MINIMA)
    ganho = numeroBrasileiro(parte2.GANHO_MAXIMO)

    return {
        CANAL_ESCURO: {
            "origem": "O canal escuro (dark channel prior) foi proposto por Kaiming He, Jian Sun "
            "e Xiaoou Tang em 2009, para tirar a névoa de fotos feitas ao ar livre. O artigo "
            "ganhou o prêmio de melhor artigo da CVPR, uma das principais conferências de visão "
            "computacional. Em 2013, Paulo Drews Jr. e colegas, pesquisadores brasileiros, "
            "adaptaram a ideia para debaixo d'água usando só os canais verde e azul, e essa "
            "versão é o UDCP. A erosão usada no cálculo vem da morfologia matemática, criada por "
            "Georges Matheron e Jean Serra na França, nos anos 1960.",
            "importancia": "Foi uma das primeiras formas simples de estimar a névoa a partir de "
            "uma única foto, sem equipamento extra nem várias imagens da mesma cena. Virou base "
            "para dezenas de métodos de remoção de névoa e de restauração subaquática.",
            "usoNoProjeto": "A Parte 1 precisa saber quanto véu de água existe em cada ponto da "
            "foto, e o UDCP dá essa medida a partir de uma imagem só. É o método que Wang et al. "
            "(2019) descrevem para imagens subaquáticas.",
        },
        COR_DA_AGUA: {
            "origem": "A ideia de uma luz de fundo que se soma à cena vem do modelo de "
            "visibilidade de Harald Koschmieder, de 1924. Nos métodos de canal escuro, essa cor "
            "se chama luz atmosférica, e a regra de pegar os 0,1% pixels mais claros do canal "
            "escuro foi proposta por He et al. (2009). Restringir a busca a uma região de fundo "
            "é uma melhoria de trabalhos posteriores, revisados por Wang et al. (2019).",
            "importancia": "A cor da água entra em todas as contas seguintes, então um erro nela "
            "estraga a restauração inteira. Por isso a estimativa dessa cor é um dos pontos mais "
            "estudados nos métodos de restauração.",
            "usoNoProjeto": "Nos testes de outubro, a regra original escolheu o cilindro branco "
            "do mergulhador como cor da água. Buscar só no terço superior e ignorar os pixels "
            "quase brancos resolveu o problema.",
        },
        NORMALIZACAO: {
            "origem": "Dividir a imagem pela luz atmosférica é um passo da dedução da "
            "transmissão no artigo de He et al. (2009). A mesma divisão aparece na versão "
            "subaquática de Drews Jr. et al. (2013), com a cor da água no lugar da luz "
            "atmosférica.",
            "importancia": "Sem essa divisão, a medida do véu dependeria da cor da água de cada "
            "foto. Com ela, a mesma régua, de 0 a 1, serve para qualquer imagem.",
            "usoNoProjeto": "É o que permite transformar o canal escuro em transmissão na etapa "
            "seguinte. O cálculo da transmissão usado no projeto, o trecho T6 da proposta, já "
            "vem com a imagem dividida pela cor da água.",
        },
        TRANSMISSAO: {
            "origem": "A ideia de que a luz enfraquece de forma exponencial ao atravessar um meio "
            "vem da lei de Beer-Lambert, construída por Pierre Bouguer (1729), Johann Lambert "
            "(1760) e August Beer (1852). A forma de estimar a transmissão a partir do canal "
            "escuro foi proposta por He et al. (2009) e adaptada para a água por Drews Jr. et "
            "al. (2013).",
            "importancia": "A transmissão funciona como um mapa de profundidade obtido de uma "
            "foto só, sem sensor de distância. É ela que diz quanto corrigir em cada ponto da "
            "imagem.",
            "usoNoProjeto": "A fórmula de McGlamery-Jaffe precisa da transmissão em cada pixel, e "
            "o UDCP é a forma de obtê-la descrita por Wang et al. (2019).",
        },
        REFINAMENTO: {
            "origem": "O filtro guiado foi proposto em 2010 por Kaiming He, Jian Sun e Xiaoou "
            "Tang, os mesmos autores do canal escuro. Ele substituiu o refinamento do artigo de "
            "2009, chamado soft matting, que dava bons resultados mas era muito lento.",
            "importancia": "Suaviza preservando as bordas e tem custo proporcional ao número de "
            "pixels, qualquer que seja o tamanho da janela. Por isso virou padrão em restauração "
            "de imagens e está disponível em bibliotecas como o OpenCV e o MATLAB.",
            "usoNoProjeto": "Wang et al. (2019) refinam todas as transmissões com ele, o trecho "
            "T7 da proposta. No projeto, ele tira os blocos da erosão sem criar halos em volta "
            "dos objetos.",
        },
        RECUPERACAO: {
            "origem": "A conta inverte o modelo de formação da imagem de McGlamery (1980) e Jaffe "
            "(1990). O piso t0 foi introduzido por He et al. (2009), que usavam t0 = 0,1.",
            "importancia": "Inverter o modelo é o que diferencia restauração de realce: em vez "
            "de só melhorar a aparência, a conta tenta desfazer o que a água fez.",
            "usoNoProjeto": "É o objetivo da Parte 1 e o centro da pergunta do projeto. O t0 = "
            f"{limite}, bem maior que o 0,1 original, foi escolhido nos testes com a imagem "
            "piloto, por dar a melhor nota à cena restaurada: pisos menores amplificam o ruído da "
            "água, e pisos maiores tiram menos véu.",
        },
        MUNDO_CINZA: {
            "origem": "O mundo cinza (gray world) foi proposto por Gershon Buchsbaum em 1980, num "
            "estudo sobre como percebemos a cor dos objetos. Ele parte da constância de cor, a "
            "capacidade do olho de reconhecer uma folha como verde tanto ao sol quanto na "
            "sombra, e supõe que a média de uma cena comum é cinza. O cinza do nome é essa "
            "suposição, e não o resultado: a imagem não fica cinza.",
            "importancia": "É um dos primeiros algoritmos de balanço de branco automático e "
            "ainda é usado como referência, por ser simples e rápido. Variações dele aparecem no "
            "balanço de branco automático de câmeras.",
            "usoNoProjeto": "Wang et al. (2019) o citam como correção de cor tradicional, e Li et "
            "al. (2019) usam balanço de branco como etapa padrão. O limite de ganho não faz parte "
            "do método original: é uma adaptação do projeto para imagens subaquáticas, porque o "
            "gray world usado direto na água distorce a cor e aumenta o ruído (Wang et al., 2019, "
            f"trecho T9). O valor {ganho} foi escolhido nos testes para evitar a cena rosa.",
            "limitacao": "Falha quando a cena tem uma cor dominante, como fotos com muita água "
            "azul ou verde: o método não consegue separar a cor que a água colocou da cor real da "
            "cena. Quando o vermelho é quase ausente, o ganho dele explode, e é isso que o limite "
            "segura.",
        },
        CLAHE: {
            "origem": "A equalização de histograma é uma técnica clássica de processamento de "
            "imagens. A versão adaptativa, que equaliza por regiões, e o limite de contraste "
            "foram descritos por Stephen Pizer e colegas em 1987, para imagens médicas. A "
            "implementação mais usada do CLAHE foi publicada por Karel Zuiderveld em 1994. O "
            "espaço de cor LAB, onde ele é aplicado, foi definido pela CIE, a Comissão "
            "Internacional de Iluminação, em 1976.",
            "importancia": "Melhora o contraste local sem amplificar demais o ruído, e por isso é "
            "muito usado em imagens médicas, de satélite e subaquáticas. Já vem pronto no "
            "OpenCV.",
            "usoNoProjeto": "As duas referências do projeto, Wang et al. e Li et al. (2019), usam "
            "equalização de histograma. A versão com limite foi escolhida porque a equalização "
            "global revelou os blocos da compressão JPEG nos testes.",
            "limitacao": "Mexe só no brilho, então não corrige a cor da água, que é tarefa da "
            "etapa 4. Também não sabe quanto véu cada ponto tem: a regra depende só do brilho, "
            "então um objeto perto e outro longe são tratados do mesmo jeito. Não recupera "
            "detalhe perdido, só estica o que existe, inclusive o ruído, ainda que com teto. E o "
            "limite e a grade são fixos para todas as imagens, mas o melhor valor pode mudar de "
            "uma imagem para outra.",
        },
        HOMOMORFICA: {
            "origem": "A filtragem homomórfica foi criada por Alan Oppenheim, Ronald Schafer e "
            "Thomas Stockham em 1968, para separar sinais que se multiplicam. Stockham a aplicou "
            "a imagens em 1972, separando iluminação e refletância. Ela usa a transformada de "
            "Fourier, de Joseph Fourier (1822), calculada pelo algoritmo rápido FFT, de Cooley e "
            "Tukey (1965).",
            "importancia": "Corrige a iluminação desigual e realça os detalhes ao mesmo tempo, "
            "com uma única operação no domínio da frequência.",
            "usoNoProjeto": "Wang et al. (2019) citam o uso dela por Prabhakar et al. para "
            "corrigir a iluminação não uniforme em imagens subaquáticas, o trecho T10 da "
            "proposta.",
        },
    }


def imagensCanalEscuro(fluxo, resultados):
    return resultados["crua"], "imagem crua", resultados["canalEscuro"], "canal escuro"


def imagensCorDaAgua(fluxo, resultados):
    return (
        resultados["canalEscuro"],
        "canal escuro, onde a cor da água é procurada",
        marcarPosicao(resultados["crua"], resultados["posicaoDaAgua"]),
        "pixel escolhido como cor da água",
    )


def imagensNormalizacao(fluxo, resultados):
    return (
        resultados["crua"],
        "imagem crua",
        resultados["normalizada"],
        "imagem dividida pela cor da água",
    )


def imagensTransmissao(fluxo, resultados):
    return (
        resultados["normalizada"],
        "imagem dividida pela cor da água",
        resultados["transmissao"],
        "transmissão t",
    )


def imagensRefinamento(fluxo, resultados):
    return (
        resultados["transmissao"],
        "transmissão com blocos",
        resultados["transmissaoRefinada"],
        "transmissão refinada",
    )


def imagensRecuperacao(fluxo, resultados):
    return resultados["crua"], "imagem crua", resultados["restaurada"], "cena restaurada J"


def imagensMundoCinza(fluxo, resultados):
    estagios = resultados[fluxo]

    return estagios["entrada"], "imagem crua", estagios["cor"], "depois do mundo cinza"


def imagensClahe(fluxo, resultados):
    estagios = resultados[fluxo]

    return estagios["cor"], "depois do mundo cinza", estagios["contraste"], "depois do CLAHE"


def imagensHomomorfica(fluxo, resultados):
    estagios = resultados[fluxo]

    return (
        estagios["contraste"],
        "depois do CLAHE",
        estagios["final"],
        "depois da filtragem homomórfica",
    )


def detalhesCanalEscuro(fluxo, resultados):
    esquerda, direita = st.columns(2)
    esquerda.image(
        parte1.minimoEntreVerdeEAzul(resultados["crua"]),
        caption="Passo intermediário, ainda não é o canal escuro: mínimo entre G e B",
        clamp=True,
    )
    direita.image(
        recorteAmpliado(resultados["canalEscuro"], centroDaImagem(resultados["canalEscuro"])),
        caption="Zoom no canal escuro: os blocos da erosão",
        clamp=True,
    )


def detalhesCorDaAgua(fluxo, resultados):
    canalEscuro = resultados["canalEscuro"]
    candidatos = parte1.mascaraDeCandidatos(resultados["crua"])
    linhas, colunas = parte1.posicoesMaisClaras(canalEscuro, candidatos)
    vermelho, verde, azul = resultados["corDaAgua"]
    fracao = numeroBrasileiro(parte1.FRACAO_MAIS_CLAROS * 100)
    primeira, segunda, terceira = st.columns(3)
    primeira.image(
        candidatos.astype(np.float64),
        caption="Área de busca: terço superior sem os pixels quase brancos",
    )
    segunda.image(
        destacarPosicoes(canalEscuro, linhas, colunas),
        caption=f"Em vermelho, os {fracao}% mais claros do canal escuro",
        clamp=True,
    )
    terceira.image(
        amostraDeCor(resultados["corDaAgua"]),
        caption=f"Cor da água: R {vermelho:.2f}, G {verde:.2f}, B {azul:.2f}",
        clamp=True,
    )


def detalhesRefinamento(fluxo, resultados):
    centro = centroDaImagem(resultados["transmissao"])
    primeira, segunda, terceira = st.columns(3)
    primeira.image(
        parte1.calcularIntensidade(resultados["crua"]),
        caption="Guia: intensidade da imagem original",
        clamp=True,
    )
    segunda.image(
        recorteAmpliado(resultados["transmissao"], centro),
        caption="Zoom antes do filtro",
        clamp=True,
    )
    terceira.image(
        recorteAmpliado(resultados["transmissaoRefinada"], centro),
        caption="Zoom depois do filtro",
        clamp=True,
    )


def detalhesRecuperacao(fluxo, resultados):
    refinada = resultados["transmissaoRefinada"]
    limite = resultados["parametros"]["piso"]
    abaixoDoLimite = np.mean(refinada < limite) * 100
    st.write(
        f"Nesta imagem, {abaixoDoLimite:.0f}% dos pixels ficam com transmissão abaixo de t0. "
        "Nesses pixels o divisor é o próprio t0, então o mapa de transmissão não muda nada e a "
        "restauração vira uma ampliação uniforme em torno da cor da água."
    )
    esquerda, direita = st.columns(2)
    esquerda.image(
        parte1.limitarTransmissao(refinada, limite),
        caption="O divisor usado em cada pixel: a transmissão, nunca abaixo do piso t0",
        clamp=True,
    )
    direita.image(
        (refinada < limite).astype(np.float64),
        caption="Em branco, os pixels em que t ficou abaixo de t0",
    )


def detalhesClahe(fluxo, resultados):
    estagios = resultados[fluxo]
    st.line_chart(
        {
            "antes": histogramaDaLuminosidade(estagios["cor"]),
            "depois": histogramaDaLuminosidade(estagios["contraste"]),
        },
        x_label="nível de L",
        y_label="quantidade de pixels",
    )
    equalizada = equalizacaoGlobal(estagios["cor"])
    regiao = regiaoDaAgua(estagios["cor"])
    primeira, segunda, terceira = st.columns(3)
    primeira.image(equalizada, caption="Se fosse equalização global", clamp=True)
    segunda.image(
        recorteAmpliado(equalizada, regiao), caption="Zoom na água: equalização global", clamp=True
    )
    terceira.image(
        recorteAmpliado(estagios["contraste"], regiao), caption="Zoom na água: CLAHE", clamp=True
    )


def detalhesHomomorfica(fluxo, resultados):
    estagios = resultados[fluxo]
    altura, largura = estagios["contraste"].shape[:2]
    parametros = resultados["parametros"]
    baixa = numeroBrasileiro(parametros["ganhoBaixas"])
    alta = numeroBrasileiro(parametros["ganhoAltas"])
    esquerda, direita = st.columns(2)
    esquerda.image(
        espectroLogaritmico(estagios["contraste"]),
        caption="Espectro do brilho: o centro são as baixas frequências",
        clamp=True,
    )
    direita.image(
        vistaDoFiltro(altura, largura, parametros),
        caption=f"Filtro perto do centro: escuro é {baixa} e claro é {alta}",
        clamp=True,
    )
    st.line_chart(
        {"ganho do filtro": perfilDoFiltro(altura, largura, parametros)},
        x_label="distância ao centro do espectro",
        y_label="ganho",
    )


def imagensDasEtapas():
    return {
        CANAL_ESCURO: imagensCanalEscuro,
        COR_DA_AGUA: imagensCorDaAgua,
        NORMALIZACAO: imagensNormalizacao,
        TRANSMISSAO: imagensTransmissao,
        REFINAMENTO: imagensRefinamento,
        RECUPERACAO: imagensRecuperacao,
        MUNDO_CINZA: imagensMundoCinza,
        CLAHE: imagensClahe,
        HOMOMORFICA: imagensHomomorfica,
    }


def detalhesDasEtapas():
    return {
        CANAL_ESCURO: detalhesCanalEscuro,
        COR_DA_AGUA: detalhesCorDaAgua,
        REFINAMENTO: detalhesRefinamento,
        RECUPERACAO: detalhesRecuperacao,
        CLAHE: detalhesClahe,
        HOMOMORFICA: detalhesHomomorfica,
    }


def conclusaoDoMundoCinza(ganhosPuros, ganhoMaximo):
    limite = numeroBrasileiro(ganhoMaximo)
    cortados = [
        f"o {nome} seria multiplicado por {numeroBrasileiro(round(ganhoPuro, 2))}"
        for nome, ganhoPuro in zip(NOMES_DOS_CANAIS, ganhosPuros)
        if ganhoPuro > ganhoMaximo
    ]

    if not cortados:
        return f"Com o limite {limite}, nenhum canal é cortado: todos os ganhos já cabem nele."

    return f"Sem o limite, {' e '.join(cortados)}. Com o limite, nenhum canal passa de {limite}."


def ganhosSemLimite(imagem):
    medias = parte2.mediasDosCanais(imagem)

    return medias.mean() / medias


def chaveDoParametro(nome):
    return f"parametro::{nome}"


def chaveDoControle(nome):
    return f"controle::{nome}"


def valorDoParametro(nome):
    return st.session_state.get(chaveDoParametro(nome), PARAMETROS_DO_PROJETO[nome])


def parametrosAtuais():
    return {nome: valorDoParametro(nome) for nome in PARAMETROS_DO_PROJETO}


def guardarParametro(nome):
    st.session_state[chaveDoParametro(nome)] = st.session_state[chaveDoControle(nome)]


def parametrosAlterados(parametros):
    return [
        nome
        for nome, valor in parametros.items()
        if not np.isclose(valor, PARAMETROS_DO_PROJETO[nome])
    ]


def voltarAosValoresDoProjeto():
    for nome in PARAMETROS_DO_PROJETO:
        st.session_state.pop(chaveDoParametro(nome), None)
        st.session_state.pop(chaveDoControle(nome), None)


def controleDeslizante(local, nome, minimo, maximo, passo):
    return local.slider(
        NOMES_DOS_PARAMETROS[nome],
        minimo,
        maximo,
        valorDoParametro(nome),
        passo,
        key=chaveDoControle(nome),
        on_change=guardarParametro,
        args=(nome,),
    )


def controleDeOpcoes(local, nome, opcoes, formatar):
    return local.select_slider(
        NOMES_DOS_PARAMETROS[nome],
        opcoes,
        value=valorDoParametro(nome),
        format_func=formatar,
        key=chaveDoControle(nome),
        on_change=guardarParametro,
        args=(nome,),
    )


def nomeDaJanela(janela):
    return f"{janela} × {janela}"


def textoDoParametro(nome, valor):
    formatos = {"janela": nomeDaJanela, "gradeClahe": nomeDaGrade, "limiteClahe": nomeDoLimite}

    return formatos.get(nome, numeroBrasileiro)(valor)


def valorDoProjeto(nome):
    return textoDoParametro(nome, PARAMETROS_DO_PROJETO[nome])


def tabelaDeParametros(parametros):
    return {
        "parâmetro": [NOMES_DOS_PARAMETROS[nome] for nome in parametros],
        "valor do projeto": [valorDoProjeto(nome) for nome in parametros],
        "valor usado agora": [textoDoParametro(nome, valor) for nome, valor in parametros.items()],
    }


def decimalBrasileiro(valor, casas):
    return f"{valor:.{casas}f}".replace(".", ",")


def mostrarEfeitoNoResultado(resultados):
    notas = resultados["notas"]
    partes = [
        f"{rotulo}: {decimalBrasileiro(notas[rotulo][0], 2)} dB e "
        f"SSIM {decimalBrasileiro(notas[rotulo][1], 3)}"
        for rotulo in (FLUXO_TRADICIONAL, FLUXO_MCGLAMERY, COMBINACAO)
    ]
    st.markdown("**Notas contra a referência com esses valores.** " + " · ".join(partes))


def demonstracaoCanalEscuro(fluxo, resultados):
    st.markdown("**Experimente: mude a janela do canal escuro**")
    st.caption(
        f"O projeto usa {valorDoProjeto('janela')}. Com uma janela maior, é mais fácil achar uma "
        "sombra por perto, e os blocos da transmissão ficam maiores. A mesma janela vale para o "
        "canal escuro da etapa 3b."
    )
    janela = controleDeOpcoes(st, "janela", JANELAS_DA_DEMONSTRACAO, nomeDaJanela)
    esquerda, direita = st.columns(2)
    esquerda.image(
        resultados["canalEscuro"],
        caption=f"Canal escuro, janela {nomeDaJanela(janela)}",
        clamp=True,
    )
    direita.image(
        resultados["transmissaoRefinada"],
        caption="Transmissão refinada com essa janela",
        clamp=True,
    )
    mostrarEfeitoNoResultado(resultados)


def demonstracaoRefinamento(fluxo, resultados):
    st.markdown("**Experimente: mude o raio do filtro guiado**")
    st.caption(
        f"O projeto usa {valorDoProjeto('raio')}. Um raio maior suaviza uma região maior. Um raio "
        "menor deixa sobrar parte dos blocos."
    )
    raio = controleDeslizante(st, "raio", 5, 60, 5)
    centro = centroDaImagem(resultados["transmissao"])
    esquerda, direita = st.columns(2)
    esquerda.image(
        recorteAmpliado(resultados["transmissao"], centro),
        caption="Zoom antes do filtro",
        clamp=True,
    )
    direita.image(
        recorteAmpliado(resultados["transmissaoRefinada"], centro),
        caption=f"Zoom depois do filtro, raio {raio}",
        clamp=True,
    )
    mostrarEfeitoNoResultado(resultados)


def demonstracaoRecuperacao(fluxo, resultados):
    st.markdown("**Experimente: mude o piso da transmissão**")
    st.caption(
        f"O projeto usa {valorDoProjeto('piso')}. Com 1, a restauração não faz nada, e o Fluxo 2 "
        "devolve a própria foto crua. Pisos menores seguem mais o mapa de transmissão, mas "
        "amplificam o ruído da água."
    )
    piso = controleDeslizante(st, "piso", 0.1, 1.0, 0.05)
    restaurada = resultados["restaurada"]
    abaixoDoPiso = np.mean(resultados["transmissaoRefinada"] < piso) * 100
    esquerda, direita = st.columns(2)
    esquerda.image(
        restaurada, caption=f"Cena restaurada, piso {numeroBrasileiro(piso)}", clamp=True
    )
    direita.image(
        recorteAmpliado(restaurada, regiaoDaAgua(restaurada)),
        caption="Zoom na água: com pisos pequenos, o ruído aparece",
        clamp=True,
    )
    st.write(f"Com esse piso, {abaixoDoPiso:.0f}% dos pixels usam o próprio piso na conta.")
    mostrarEfeitoNoResultado(resultados)


def demonstracaoMundoCinza(fluxo, resultados):
    entrada = resultados[fluxo]["entrada"]
    st.markdown("**Experimente: mude o limite do ganho**")
    st.caption(
        f"O projeto usa {valorDoProjeto('ganhoMaximo')}. O valor vale para o Fluxo 1 e para "
        f"{COMBINACAO}."
    )
    ganhoMaximo = controleDeslizante(st, "ganhoMaximo", 1.0, 4.0, 0.1)
    esquerda, direita = st.columns(2)
    esquerda.image(
        resultados[fluxo]["cor"],
        caption=f"Depois, com limite {numeroBrasileiro(ganhoMaximo)}",
        clamp=True,
    )
    with direita:
        st.dataframe(tabelaDeGanhos(entrada, ganhoMaximo), hide_index=True)
        st.write(conclusaoDoMundoCinza(ganhosSemLimite(entrada), ganhoMaximo))
    mostrarEfeitoNoResultado(resultados)


def demonstracaoHomomorfica(fluxo, resultados):
    estagios = resultados[fluxo]
    altura, largura = estagios["contraste"].shape[:2]
    st.markdown("**Experimente: mude o filtro homomórfico**")
    st.caption(
        f"O projeto usa ganho {valorDoProjeto('ganhoBaixas')} na iluminação, "
        f"{valorDoProjeto('ganhoAltas')} nos detalhes e corte {valorDoProjeto('corte')}. Ganho "
        "menor na iluminação deixa a luz mais uniforme. Ganho maior nos detalhes realça as "
        "texturas, e o ruído junto. Os valores valem para o Fluxo 1 e para "
        f"{COMBINACAO}."
    )
    primeira, segunda, terceira = st.columns(3)
    controleDeslizante(primeira, "ganhoBaixas", 0.3, 1.0, 0.05)
    controleDeslizante(segunda, "ganhoAltas", 1.0, 2.0, 0.05)
    controleDeslizante(terceira, "corte", 5, 100, 5)
    esquerda, direita = st.columns(2)
    esquerda.image(estagios["final"], caption="Depois, com esses valores", clamp=True)
    direita.image(
        vistaDoFiltro(altura, largura, resultados["parametros"]),
        caption="O filtro perto do centro do espectro: escuro atenua, claro realça",
        clamp=True,
    )
    mostrarEfeitoNoResultado(resultados)


def nomeDoLimite(limite):
    return "sem limite" if limite == SEM_LIMITE else numeroBrasileiro(limite)


def nomeDaGrade(blocos):
    return f"{blocos} × {blocos}"


def inteiroBrasileiro(valor):
    return f"{valor:,}".replace(",", ".")


def conclusaoDoClahe(esticamento, blocos):
    vezes = numeroBrasileiro(round(esticamento, 1))
    if esticamento > ESTICAMENTO_EXAGERADO:
        conclusao = (
            f"O CLAHE estica os tons da água até {vezes} vezes. Diferenças mínimas, como o ruído "
            "e os quadradinhos da compressão JPEG, viram manchas. Repare no zoom."
        )
    else:
        conclusao = (
            f"Com o teto, o CLAHE estica os tons da água no máximo {vezes} vezes. Quanto menor "
            "esse número, mais discreto fica o ruído da água."
        )
    if blocos == 1:
        conclusao += (
            " Com um bloco só, a correção é a mesma para a foto inteira, como na equalização "
            "global."
        )

    return conclusao


def niveisMaisCheios(histograma, quantidade):
    ordem = np.argsort(histograma, kind="stable")[::-1][:quantidade]

    return np.sort(ordem[histograma[ordem] > 0])


def regraComLimite(analise, limite):
    pixelsNoBloco = analise["pixels"].size
    teto = tetoDoBloco(pixelsNoBloco, limite)

    return regraDeTroca(cortarHistograma(analise["histograma"], teto), pixelsNoBloco)


def tabelaDoBloco(analise, limite):
    niveis = niveisMaisCheios(analise["histograma"], LINHAS_DA_TABELA)
    tabela = {"nível de L": niveis, "pixels": analise["histograma"][niveis]}
    if limite != SEM_LIMITE:
        tabela["depois do corte"] = analise["cortado"][niveis]
        comLimite = regraComLimite(analise, limite)[niveis].astype(int)
        tabela[f"vira, com limite {nomeDoLimite(limite)}"] = comLimite
    tabela["vira, sem limite"] = regraComLimite(analise, SEM_LIMITE)[niveis].astype(int)

    return tabela


def mostrarTabelaDoBloco(analise, limite):
    st.markdown("**O bloco do exemplo, nível por nível**")
    st.dataframe(tabelaDoBloco(analise, limite), hide_index=True)
    st.caption("Os níveis com mais pixels no bloco. 'Vira' é o novo brilho pela regra do bloco.")


def niveisDoGrafico(pixels):
    inicio, fim = np.percentile(pixels, PERCENTIS_DO_GRAFICO)
    primeiro = max(int(inicio) - MARGEM_DO_GRAFICO, 0)
    ultimo = min(int(fim) + MARGEM_DO_GRAFICO, NIVEIS_DE_CINZA - 1)

    return np.arange(primeiro, ultimo + 1)


def rotuloDoEixo(valor, posicao):
    return inteiroBrasileiro(int(valor))


def desenharBarrasEmPilha(eixo, niveis, base, topo, corDoTopo):
    eixo.bar(niveis, base, width=LARGURA_DA_BARRA, color=COR_ABAIXO_DO_TETO)
    eixo.bar(niveis, topo, width=LARGURA_DA_BARRA, bottom=base, color=corDoTopo)


def desenharTeto(eixo, teto):
    eixo.axhline(teto, color=COR_DO_TETO, linestyle="--", linewidth=1.5)


def formatarEixoDoCorte(eixo, titulo):
    eixo.set_title(titulo, fontsize=10)
    eixo.set_xlabel("nível de L", fontsize=9)
    eixo.set_ylabel("pixels no bloco", fontsize=9)
    eixo.xaxis.set_major_locator(MaxNLocator(integer=True))
    eixo.yaxis.set_major_formatter(FuncFormatter(rotuloDoEixo))
    eixo.tick_params(labelsize=8)
    eixo.spines[["top", "right"]].set_visible(False)


def legendaDoCorte(teto):
    rotuloDoTeto = f"teto de {inteiroBrasileiro(teto)} pixels"

    return [
        Patch(color=COR_ABAIXO_DO_TETO, label="abaixo do teto"),
        Patch(color=COR_DO_EXCESSO, label="excesso, cortado"),
        Patch(color=COR_DEVOLVIDA, label="parte do excesso devolvida"),
        Line2D([], [], color=COR_DO_TETO, linestyle="--", label=rotuloDoTeto),
    ]


def painelAntesDoCorte(eixo, analise, niveis):
    histograma = analise["histograma"][niveis]
    abaixo = np.minimum(histograma, analise["teto"])
    desenharBarrasEmPilha(eixo, niveis, abaixo, histograma - abaixo, COR_DO_EXCESSO)
    desenharTeto(eixo, analise["teto"])
    formatarEixoDoCorte(eixo, "Antes do corte")


def painelDepoisDoCorte(eixo, analise, niveis):
    abaixo = np.minimum(analise["histograma"][niveis], analise["teto"])
    devolvido = analise["cortado"][niveis] - abaixo
    desenharBarrasEmPilha(eixo, niveis, abaixo, devolvido, COR_DEVOLVIDA)
    desenharTeto(eixo, analise["teto"])
    formatarEixoDoCorte(eixo, "Depois do corte (outra escala)")


def painelSemTeto(eixo, analise, niveis):
    histograma = analise["histograma"][niveis]
    eixo.bar(niveis, histograma, width=LARGURA_DA_BARRA, color=COR_ABAIXO_DO_TETO)
    formatarEixoDoCorte(eixo, "Sem limite: não há teto, nada é cortado")


def graficoDoCorte(analise, limite):
    niveis = niveisDoGrafico(analise["pixels"])
    figura = Figure(figsize=TAMANHO_DO_GRAFICO, layout="constrained")
    if limite == SEM_LIMITE:
        painelSemTeto(figura.subplots(), analise, niveis)
        return figura
    antes, depois = figura.subplots(1, 2)
    painelAntesDoCorte(antes, analise, niveis)
    painelDepoisDoCorte(depois, analise, niveis)
    figura.legend(
        handles=legendaDoCorte(analise["teto"]),
        loc="outside lower center",
        ncols=2,
        fontsize=8,
        frameon=False,
    )

    return figura


def mostrarGraficoDoCorte(analise, limite):
    st.markdown("**O corte no teto, em barras**")
    st.pyplot(graficoDoCorte(analise, limite))
    if limite != SEM_LIMITE:
        st.caption(
            "Azul: o que fica abaixo do teto. Vermelho: o excesso, que é cortado. Verde: a parte "
            "do excesso que cada nível recebe de volta. Os dois painéis mostram só os níveis em "
            "que o bloco tem pixels, e cada um tem a sua escala."
        )


def demonstracaoClahe(fluxo, resultados):
    entrada = resultados[fluxo]["cor"]
    luminosidade = np.ascontiguousarray(luminosidadeDe(entrada))
    st.markdown("**Experimente: mude o limite e a grade**")
    st.caption(
        f"O projeto usa limite {valorDoProjeto('limiteClahe')} e grade "
        f"{valorDoProjeto('gradeClahe')}. Escolha 'sem limite' para ver a água virar manchas, e a "
        "grade 1 × 1 para usar a mesma correção na foto inteira. Os valores valem para o Fluxo 1 "
        f"e para {COMBINACAO}."
    )
    esquerda, direita = st.columns(2)
    limite = controleDeOpcoes(esquerda, "limiteClahe", LIMITES_DA_DEMONSTRACAO, nomeDoLimite)
    blocos = controleDeOpcoes(direita, "gradeClahe", GRADES_DA_DEMONSTRACAO, nomeDaGrade)
    saida = resultados[fluxo]["contraste"]
    analise = analisarBloco(luminosidade, limite, blocos)
    esticamento = esticamentoMaximo(analise["cortado"], analise["pixels"].size)
    grade = desenharGrade(
        entrada, analise["alturaBloco"], analise["larguraBloco"], analise["bloco"]
    )

    primeira, segunda, terceira = st.columns(3)
    primeira.image(
        grade, caption="A grade de blocos. Em vermelho, o bloco do exemplo.", clamp=True
    )
    segunda.image(
        saida,
        caption=f"Depois, com limite {nomeDoLimite(limite)} e grade {nomeDaGrade(blocos)}",
        clamp=True,
    )
    terceira.image(
        recorteAmpliado(saida, regiaoDaAgua(saida)),
        caption="Zoom na água com esses valores",
        clamp=True,
    )
    st.write(conclusaoDoClahe(esticamento, blocos))
    esquerda, direita = st.columns(2)
    with esquerda:
        mostrarGraficoDoCorte(analise, limite)
    with direita:
        mostrarTabelaDoBloco(analise, limite)
    mostrarEfeitoNoResultado(resultados)


def demonstracoesDasEtapas():
    return {
        CANAL_ESCURO: demonstracaoCanalEscuro,
        REFINAMENTO: demonstracaoRefinamento,
        RECUPERACAO: demonstracaoRecuperacao,
        MUNDO_CINZA: demonstracaoMundoCinza,
        CLAHE: demonstracaoClahe,
        HOMOMORFICA: demonstracaoHomomorfica,
    }


def pixelEmLab(pixel):
    return parte2.rgbParaLab(pixel.reshape(1, 1, 3)).reshape(3).astype(int)


def labEmPixel(lab):
    lab8 = np.clip(lab, 0, utilitarios.INTENSIDADE_MAXIMA).astype(np.uint8).reshape(1, 1, 3)

    return parte2.labParaRgb(lab8).reshape(3)


def labPedido(lab, mudanca):
    return np.clip(lab + np.array(mudanca), 0, utilitarios.INTENSIDADE_MAXIMA)


def pixelComMudanca(pixel, mudanca):
    if not any(mudanca):
        return pixel

    return labEmPixel(labPedido(pixelEmLab(pixel), mudanca))


def niveisDoPixel(pixel):
    return utilitarios.paraNiveisDeCinza(pixel).astype(int)


def labComSinal(lab):
    return lab - np.array([0, DESLOCAMENTO_LAB, DESLOCAMENTO_LAB])


def textoDoTrio(valores):
    return ", ".join(str(int(valor)).replace("-", "−") for valor in valores)


def comMaiuscula(texto):
    return texto[0].upper() + texto[1:]


def direcaoDoCanal(diferenca):
    if diferenca > TOLERANCIA_DE_MUDANCA:
        return "subiu"
    if diferenca < -TOLERANCIA_DE_MUDANCA:
        return "desceu"

    return "ficou quase igual"


def descreverCanaisRgb(antes, depois):
    direcoes = [direcaoDoCanal(int(novo) - int(velho)) for velho, novo in zip(antes, depois)]
    if direcoes == ["subiu"] * 3:
        return "os três canais do RGB subiram juntos"
    if direcoes == ["desceu"] * 3:
        return "os três canais do RGB desceram juntos"
    if direcoes == ["ficou quase igual"] * 3:
        return "nenhum canal do RGB mudou"
    partes = [f"o {nome} {direcao}" for nome, direcao in zip(NOMES_DOS_CANAIS, direcoes)]

    return f"{partes[0]}, {partes[1]} e {partes[2]}"


def corFoiMantida(labAntes, labDepois):
    return bool(np.all(np.abs(labDepois[1:] - labAntes[1:]) <= TOLERANCIA_DE_MUDANCA))


def descreverMudancaDoPixel(lab, pedido, pixelAntes, pixelDepois):
    canais = comMaiuscula(descreverCanaisRgb(niveisDoPixel(pixelAntes), niveisDoPixel(pixelDepois)))
    if corFoiMantida(lab, pixelEmLab(pixelDepois)):
        return f"{canais}, e a cor (a e b) continua a mesma."
    if corFoiMantida(lab, pedido):
        return (
            f"{canais}. A cor mudou um pouco só porque algum canal chegou ao limite de 0 ou 255 "
            "e foi cortado."
        )

    return f"{canais}, e a cor mudou."


def linhaDaTabela(celulas):
    return "| " + " | ".join(celulas) + " |"


def tabelaDoExemploLab(pixel):
    lab = pixelEmLab(pixel)
    linhas = [
        linhaDaTabela(["O que mudou no LAB", "L, a, b", "R, G, B", "O que aconteceu"]),
        linhaDaTabela(["---"] * 4),
    ]
    for nome, mudanca in MUDANCAS_DO_EXEMPLO:
        pedido = labPedido(lab, mudanca)
        novo = pixelComMudanca(pixel, mudanca)
        descricao = "o pixel como está"
        if any(mudanca):
            descricao = descreverMudancaDoPixel(lab, pedido, pixel, novo)
        celulas = [nome, textoDoTrio(labComSinal(pedido)), textoDoTrio(niveisDoPixel(novo))]
        linhas.append(linhaDaTabela(celulas + [descricao]))

    return "\n".join(linhas)


def amostrasDoExemploLab(pixel):
    return np.hstack(
        [amostraDeCor(pixelComMudanca(pixel, mudanca)) for _, mudanca in MUDANCAS_DO_EXEMPLO]
    )


def folhasDoLab(imagem):
    lab = parte2.rgbParaLab(imagem).astype(np.float64)
    folhaL = lab[:, :, parte2.LUMINOSIDADE] / utilitarios.INTENSIDADE_MAXIMA

    return folhaL, normalizarParaExibicao(lab[:, :, 1]), normalizarParaExibicao(lab[:, :, 2])


def tabelaDosCanaisLab():
    return "\n".join(
        [
            linhaDaTabela(["Folha", "Pergunta que responde", "Valores"]),
            linhaDaTabela(["---"] * 3),
            linhaDaTabela(["L", "quão claro é o pixel?", "de 0 (preto) a 255 (branco)"]),
            linhaDaTabela(
                ["a", "puxa para o verde ou o vermelho?", "negativo é verde, positivo é vermelho"]
            ),
            linhaDaTabela(
                ["b", "puxa para o azul ou o amarelo?", "negativo é azul, positivo é amarelo"]
            ),
        ]
    )


def mostrarFolhasDoLab(entrada, centro):
    folhaL, folhaA, folhaB = folhasDoLab(entrada)
    primeira, segunda, terceira, quarta = st.columns(4)
    primeira.image(
        marcarPosicao(entrada, centro),
        caption="Entrada em RGB. O círculo marca o pixel do exemplo.",
        clamp=True,
    )
    segunda.image(folhaL, caption="Folha L: o brilho, a foto em preto e branco", clamp=True)
    terceira.image(folhaA, caption="Folha a: escuro puxa para o verde, claro para o vermelho")
    quarta.image(folhaB, caption="Folha b: escuro puxa para o azul, claro para o amarelo")
    st.caption("As folhas a e b estão com o contraste aumentado, só para dar para ver.")


def demonstracaoDoPixelLab(pixel, fluxo):
    st.markdown("**Experimente: mude o L, o a ou o b desse pixel**")
    colunas = st.columns(3)
    mudanca = tuple(
        coluna.slider(
            f"Mudar o {canal}",
            -MUDANCA_MAXIMA_LAB,
            MUDANCA_MAXIMA_LAB,
            0,
            PASSO_MUDANCA_LAB,
            key=f"lab{canal}::{fluxo}",
        )
        for coluna, canal in zip(colunas, "Lab")
    )
    lab = pixelEmLab(pixel)
    pedido = labPedido(lab, mudanca)
    novo = pixelComMudanca(pixel, mudanca)
    esquerda, direita = st.columns(2)
    esquerda.image(
        np.hstack([amostraDeCor(pixel), amostraDeCor(novo)]),
        caption="Cor do pixel antes (esquerda) e depois (direita)",
        clamp=True,
    )
    with direita:
        st.markdown(
            f"Antes: L, a, b = {textoDoTrio(labComSinal(lab))} → "
            f"R, G, B = {textoDoTrio(niveisDoPixel(pixel))}"
        )
        st.markdown(
            f"Depois: L, a, b = {textoDoTrio(labComSinal(pedido))} → "
            f"R, G, B = {textoDoTrio(niveisDoPixel(novo))}"
        )
        if any(mudanca):
            st.write(descreverMudancaDoPixel(lab, pedido, pixel, novo))
        else:
            st.write("Mova um dos controles para ver o que acontece com o RGB.")


def explicacaoDoLab(fluxo, resultados):
    entrada = resultados[fluxo]["cor"]
    centro = centroDaImagem(entrada)
    linha, coluna = centro
    pixel = entrada[centro]

    with st.container(border=True):
        st.markdown("**Para entender o CLAHE: o que é o L do LAB**")
        st.write(
            "O RGB descreve a cor pela mistura de vermelho, verde e azul, que é como a câmera e a "
            "tela fabricam a cor. Nele, o brilho e a cor estão misturados nos três canais. O LAB, "
            "definido pela CIE em 1976, traduz cada pixel para três números novos, todos "
            "calculados a partir de R, G e B. Depois da conversão, a imagem fica guardada como "
            "três folhas, L, a e b, e não existem mais as folhas R, G e B. O CLAHE mexe só na "
            "folha L."
        )
        st.markdown(tabelaDosCanaisLab())
        st.write(
            "As três folhas são independentes. Um verde e um vermelho podem ter o mesmo L, e um "
            "verde-escuro e um verde-claro têm o mesmo a e b. O L não carrega a cor, e o a e o b "
            "não carregam o brilho."
        )
        mostrarFolhasDoLab(entrada, centro)
        st.markdown("**Exemplo pronto: um pixel da água**")
        st.write(
            f"O pixel do meio da imagem (linha {linha}, coluna {coluna}), o mesmo do Passo 4 da "
            "etapa 4, já corrigido pelo mundo cinza. Aqui os valores estão de 0 a 255, a escala "
            "que o CLAHE usa. No OpenCV, o a e o b são guardados somando 128, para caber de 0 a "
            "255. Na tabela eles aparecem com o sinal de verdade."
        )
        st.markdown(tabelaDoExemploLab(pixel))
        st.image(
            amostrasDoExemploLab(pixel),
            caption="Da esquerda para a direita: original, só o L, só o a, só o b",
            clamp=True,
        )
        demonstracaoDoPixelLab(pixel, fluxo)
        st.markdown(
            "**Conclusão.** Mexer só no L faz os três canais do RGB subirem ou descerem juntos, e "
            "a cor continua a mesma. Mexer no a ou no b muda a cor. Por isso o CLAHE equaliza só "
            "a folha L e devolve as folhas a e b sem mudança."
        )


def tabelaDosPassosDoClahe():
    blocos = parte2.BLOCOS_CLAHE
    limite = numeroBrasileiro(parte2.LIMITE_CONTRASTE_CLAHE)

    return "\n".join(
        [
            linhaDaTabela(["Passo", "O que faz", "Por que existe", "Onde ver nesta página"]),
            linhaDaTabela(["---"] * 4),
            linhaDaTabela(
                [
                    "1. Divide em blocos",
                    f"Separa a foto numa grade de {blocos}×{blocos}, ou seja, {blocos * blocos} "
                    "blocos",
                    "Cada região ganha a sua própria correção",
                    "A primeira imagem do Experimente, com a grade desenhada",
                ]
            ),
            linhaDaTabela(
                [
                    "2. Histograma de cada bloco",
                    "Conta quantos pixels há em cada nível de brilho, dentro do bloco",
                    "Mostra quais tons estão espremidos naquela região",
                    "As barras azuis do gráfico e a coluna 'pixels' da tabela",
                ]
            ),
            linhaDaTabela(
                [
                    "3. Corte no teto",
                    f"Com o limite {limite}, o que passa do teto é cortado e espalhado entre "
                    "todos os níveis",
                    "Evita esticar o ruído das regiões lisas, como a água",
                    "A parte vermelha (cortada) e a verde (devolvida) do gráfico, e a coluna "
                    "'depois do corte'",
                ]
            ),
            linhaDaTabela(
                [
                    "4. Equalização do bloco",
                    "Cada tom ganha um brilho novo, e os tons mais ocupados ganham mais espaço",
                    "É o que espalha os tons e aumenta o contraste",
                    "A coluna 'vira, com limite' da tabela",
                ]
            ),
            linhaDaTabela(
                [
                    "5. Mistura dos blocos",
                    "Cada pixel combina as correções dos blocos vizinhos",
                    "Evita que a grade apareça no resultado",
                    "A imagem 'Depois', sem emendas",
                ]
            ),
        ]
    )


def passosDoClahe(fluxo, resultados):
    with st.container(border=True):
        st.markdown("**Os passos do CLAHE**")
        st.markdown(tabelaDosPassosDoClahe())


def introducoesDasEtapas():
    return {CLAHE: [explicacaoDoLab, passosDoClahe]}


def localDoCodigo(funcao):
    linhas, inicio = inspect.getsourcelines(funcao)
    arquivo = Path(inspect.getsourcefile(funcao)).name

    return f"{arquivo}, linhas {inicio} a {inicio + len(linhas) - 1}"


def mostrarFuncao(funcao):
    st.caption(localDoCodigo(funcao))
    st.code(inspect.getsource(funcao), language="python")


def mostrarTrechoDeCodigo(funcoesPrincipais, funcoesAuxiliares):
    st.markdown("**Trecho do código**")
    for funcao in funcoesPrincipais:
        mostrarFuncao(funcao)
    if funcoesAuxiliares:
        with st.expander("Funções auxiliares"):
            for funcao in funcoesAuxiliares:
                mostrarFuncao(funcao)


def mostrarAntesDepois(antes, legendaAntes, depois, legendaDepois):
    esquerda, direita = st.columns(2)
    esquerda.image(antes, caption=f"Antes: {legendaAntes}", clamp=True)
    direita.image(depois, caption=f"Depois: {legendaDepois}", clamp=True)


def mostrarDoisQuadros(tituloEsquerda, textoEsquerda, tituloDireita, textoDireita):
    esquerda, direita = st.columns(2)
    with esquerda.container(border=True):
        st.markdown(f"**{tituloEsquerda}**")
        st.write(textoEsquerda)
    with direita.container(border=True):
        st.markdown(f"**{tituloDireita}**")
        st.write(textoDireita)


def mostrarHistoria(historia):
    with st.container(border=True):
        st.markdown(f"**Origem.** {historia['origem']}")
        st.markdown(f"**Por que é importante.** {historia['importancia']}")
        st.markdown(f"**Por que entrou no projeto.** {historia['usoNoProjeto']}")
        if "limitacao" in historia:
            st.markdown(f"**Limitação.** {historia['limitacao']}")


def mostrarSimbolos(simbolos):
    itens = [f"- ${simbolo}$: {significado}" for simbolo, significado in simbolos]
    st.markdown("**O que cada símbolo significa**")
    st.markdown("\n".join(itens))


def mostrarComoFunciona(explicacao):
    st.markdown("**Como funciona**")
    st.write(explicacao)


def parteDaEtapa(etapa):
    return PARTE_1 if etapa in ETAPAS_PARTE_1 else PARTE_2


def mostrarProgresso(fluxo, etapa):
    etapas = ETAPAS_DO_FLUXO[fluxo]
    posicao = etapas.index(etapa) + 1
    st.progress(posicao / len(etapas), text=f"{fluxo} · {posicao} de {len(etapas)}")


def mostrarEtapa(fluxo, etapa, resultados):
    textos = textosDasEtapas()[etapa]
    antes, legendaAntes, depois, legendaDepois = imagensDasEtapas()[etapa](fluxo, resultados)

    mostrarProgresso(fluxo, etapa)
    st.header(etapa)
    st.markdown(parteDaEtapa(etapa))
    mostrarHistoria(historiaDasEtapas()[etapa])
    mostrarAntesDepois(antes, legendaAntes, depois, legendaDepois)
    st.caption(f"O que observar: {textos['observar']}")
    mostrarDoisQuadros(
        "Problema que resolve", textos["problema"], "Por que usar este método", textos["motivo"]
    )
    for introducao in introducoesDasEtapas().get(etapa, []):
        introducao(fluxo, resultados)
    mostrarComoFunciona(textos["comoFunciona"])
    if etapa in demonstracoesDasEtapas():
        demonstracoesDasEtapas()[etapa](fluxo, resultados)
    mostrarTrechoDeCodigo(*CODIGO_DAS_ETAPAS[etapa])
    if etapa in detalhesDasEtapas():
        with st.expander("Ver mais detalhes"):
            detalhesDasEtapas()[etapa](fluxo, resultados)


def simularAgua(cena, corDaAgua, transmissao):
    return cena * transmissao + corDaAgua * (1 - transmissao)


def tabelaDeVariaveis():
    limite = numeroBrasileiro(parte1.TRANSMISSAO_MINIMA)

    return (
        "| Variável | O que é | Situação | Como é obtida | Etapa |\n"
        "|---|---|---|---|---|\n"
        "| I(x) | a foto captada | conhecida | leitura do arquivo | base de dados |\n"
        "| B | a cor da água, um valor por canal | desconhecida | cor dos pixels com mais véu, "
        "os mais claros do canal escuro | 1 e 2 |\n"
        "| t(x) | a transmissão, um valor por pixel | desconhecida | 1 menos o canal escuro da "
        "imagem dividida por B, refinada pelo filtro guiado | 3a, 3b e 3c |\n"
        f"| t0 | o piso da transmissão | parâmetro | escolhido nos testes com a imagem piloto "
        f"({limite}) | 3d |\n"
        "| J(x) | a cena sem água | o que queremos | invertendo a fórmula | 3d |\n"
    )


def paginaModelo(fluxo, resultados):
    mostrarProgresso(fluxo, MODELO)
    st.header(MODELO)
    st.markdown(PARTE_1)
    mostrarHistoria(HISTORIA_DO_MODELO)
    st.write(
        "O Fluxo 1 mexe direto nos pixels, sem saber como a água formou a imagem. O Fluxo 2 "
        "começa pela física: uma equação que descreve como a foto se forma debaixo d'água. Se as "
        "variáveis dessa equação forem conhecidas, dá para fazer a conta ao contrário e tirar o "
        "efeito da água. As etapas 1 a 3d existem para descobrir essas variáveis."
    )

    st.subheader("A fórmula")
    st.write(
        "A luz que chega à câmera é a soma de três partes: a luz direta do objeto, o espalhamento "
        "frontal e o retroespalhamento, que é a luz espalhada pela própria água em direção à "
        "câmera. Com a câmera perto da cena, o espalhamento frontal pode ser ignorado (Wang et "
        "al., 2019). Sobra, para cada canal de cor c, a fórmula usada no projeto:"
    )
    st.latex(
        r"I^c(x) = \underbrace{J^c(x)\,t(x)}_{\text{direta}} + "
        r"\underbrace{B^c\,\big(1 - t(x)\big)}_{\text{retroespalhamento}}"
    )
    st.write(
        "Cada pixel é uma mistura de duas cores: a do objeto (J), na proporção t, e a da água (B), "
        "na proporção 1 − t. Quanto mais longe o objeto, mais água existe no caminho e menor é a "
        "transmissão."
    )
    mostrarSimbolos(SIMBOLOS_DO_MODELO)

    st.subheader("As variáveis")
    st.markdown(tabelaDeVariaveis())
    st.write(
        "Com B e t estimados, dá para fazer a conta ao contrário e isolar J, a cena sem água. É "
        "isso que a etapa 3d faz."
    )

    st.subheader("As variáveis nesta imagem")
    primeira, segunda, terceira, quarta = st.columns(4)
    primeira.image(resultados["crua"], caption="I: a foto, conhecida", clamp=True)
    segunda.image(
        amostraDeCor(resultados["corDaAgua"]), caption="B: estimada nas etapas 1 e 2", clamp=True
    )
    terceira.image(
        resultados["transmissaoRefinada"], caption="t: estimada nas etapas 3a a 3c", clamp=True
    )
    quarta.image(resultados["restaurada"], caption="J: obtida na etapa 3d", clamp=True)

    st.subheader("O raciocínio")
    st.markdown(
        "**1. A pista: o canal escuro (etapa 1).** Uma foto sozinha não diz quanto véu existe "
        "em cada ponto. O canal escuro (dark channel) dá essa medida, e dele saem as duas "
        "variáveis que faltam."
    )
    st.markdown(
        "**2. A cor da água B (etapa 2).** Onde só há água, a transmissão é quase zero e a foto "
        "mostra a própria cor da água. Esses pixels são os de mais véu, os mais claros do canal "
        "escuro."
    )
    st.markdown(
        "**3. A transmissão t (etapas 3a a 3c).** Dividindo a foto pela cor da água, que é a "
        "etapa 3a, a água passa a valer 1. Toda região de uma cena sem água tem algum pixel "
        "escuro, então o canal escuro da imagem dividida mede quase só a água. A transmissão é 1 "
        "menos esse valor, que é a etapa 3b. Como o cálculo é feito em janelas, ela sai em "
        "blocos, e o filtro guiado remove esses blocos na etapa 3c."
    )
    st.markdown(
        "**4. A cena J (etapa 3d).** Com B e t, a fórmula é feita ao contrário. O piso t0 evita "
        "dividir por transmissões muito pequenas, que multiplicariam o ruído."
    )

    st.subheader("Experimente a fórmula")
    st.write(
        "O controle abaixo aplica a fórmula com a mesma transmissão em toda a imagem, usando a "
        "referência no papel da cena sem água J e a cor da água B desta foto. Com t = 1 não há "
        "água entre a câmera e a cena. Conforme t cai, o véu da cor da água toma conta."
    )
    transmissao = st.slider("Transmissão t", 0.0, 1.0, 0.5, 0.05)
    esquerda, direita = st.columns(2)
    esquerda.image(
        resultados["referencia"], caption="J: a referência no papel da cena sem água", clamp=True
    )
    direita.image(
        simularAgua(resultados["referencia"], resultados["corDaAgua"], transmissao),
        caption=f"I simulada com t = {numeroBrasileiro(transmissao)}",
        clamp=True,
    )
    mostrarTrechoDeCodigo(
        [parte1.recuperarCena], [parte1.limitarTransmissao, parte1.adicionarEixoDeCanal]
    )


def paginaIntroducao(resultados):
    st.title("Melhoria de imagens subaquáticas")
    st.subheader("Métodos tradicionais contra o modelo físico de McGlamery-Jaffe")
    esquerda, direita = st.columns(2)
    esquerda.image(resultados["crua"], caption="Foto subaquática do UIEB", clamp=True)
    direita.image(resultados["referencia"], caption="Referência da mesma foto", clamp=True)
    vermelho, verde, azul = resultados["crua"].mean(axis=(0, 1)).round(2)
    mostrarDoisQuadros(
        "O problema",
        "Debaixo d'água a luz é absorvida e espalhada. O vermelho some primeiro, e a imagem fica "
        "azul-esverdeada, com pouco contraste e um véu sobre o que está longe. Nesta foto, a "
        f"média do vermelho é {numeroBrasileiro(vermelho)}, contra {numeroBrasileiro(verde)} do "
        f"verde e {numeroBrasileiro(azul)} do azul.",
        "A pergunta do projeto",
        "Métodos tradicionais corrigem os sintomas sem olhar a causa. O modelo físico de "
        "McGlamery-Jaffe descreve como a água degrada a imagem e tenta desfazer essa degradação. "
        "Qual dos dois caminhos chega mais perto da referência, a imagem escolhida por pessoas "
        "como o melhor realce da foto?",
    )
    st.markdown("**O que será abordado**")
    st.markdown(
        "- **Fluxo 1 — Tradicional.** Corrige os sintomas na imagem crua: a cor com o mundo "
        "cinza (gray world), o contraste com o CLAHE e a iluminação com a filtragem "
        "homomórfica.\n"
        "- **Fluxo 2 — McGlamery.** Desfaz a causa com o modelo físico: estima o véu, a cor da "
        "água e a transmissão, e recupera a cena sem a água.\n"
        "- **Resultado.** Compara a saída de cada fluxo com a referência do UIEB, usando PSNR e "
        f"SSIM. Mostra também {COMBINACAO}, que aplica os métodos tradicionais depois da "
        "restauração.\n"
        "- **Resultados globais.** Testa combinações de parâmetros nas 890 fotos do UIEB, fica "
        "com a melhor de cada fluxo e mostra como cada método se sai na base inteira."
    )
    st.html(diagramaDosFluxos())
    st.info(
        "Os dois fluxos partem da mesma imagem crua e seguem caminhos diferentes: o Fluxo 1 só "
        "com os métodos tradicionais, em azul, e o Fluxo 2 só com a restauração física, em "
        "laranja. A comparação mostra qual caminho chega mais perto da referência."
    )
    st.markdown("**A imagem piloto e a navegação**")
    st.write(
        "As fotos vêm do UIEB (Li et al., 2019), que traz 890 imagens subaquáticas com uma "
        "referência para cada uma. O projeto usa uma imagem piloto, a 13_img_, com o mergulhador: "
        "todos os parâmetros foram escolhidos nela, e a comparação entre os fluxos é feita nela. "
        "A seção Resultados globais repete a comparação nas 890 fotos, com os valores que "
        "funcionam melhor na base inteira. "
        "Na barra lateral você escolhe a imagem e a seção. Em cada etapa aparecem o antes e o "
        "depois, o problema que a etapa resolve, o motivo da escolha do método e o trecho do "
        "código. Os botões Anterior e Próxima seguem a ordem: introdução, Fluxo 1, Fluxo 2, "
        "resultado e resultados globais. O botão Abrir slides, no alto da barra lateral, mostra "
        "os slides da apresentação um por um."
    )
    st.info(
        "Os parâmetros de cada etapa podem ser mudados nos controles 'Experimente', e valem para "
        "o fluxo daquela etapa e para o resultado. Assim dá para escolher outra foto e ver se ela "
        "fica melhor ou pior que a imagem piloto. A barra lateral avisa quando algum valor foi "
        "mudado."
    )


def paginaResultado(resultados):
    st.header("Resultado")
    st.caption(f"As saídas dos dois fluxos e de {COMBINACAO}, comparadas com a referência do UIEB")
    mostrarHistoria(HISTORIA_DA_AVALIACAO)
    primeira, segunda, terceira = st.columns(3)
    primeira.image(resultados[FLUXO_TRADICIONAL]["final"], caption=FLUXO_TRADICIONAL, clamp=True)
    segunda.image(resultados[FLUXO_MCGLAMERY]["final"], caption=FLUXO_MCGLAMERY, clamp=True)
    terceira.image(resultados[COMBINACAO]["final"], caption=COMBINACAO, clamp=True)
    esquerda, direita = st.columns(2)
    esquerda.image(resultados["crua"], caption="Imagem crua, antes de tudo", clamp=True)
    direita.image(
        resultados["referencia"],
        caption="Referência do UIEB: o controle, escolhido por voluntários",
        clamp=True,
    )
    notas = resultados["notas"]
    mostrarGraficosDasNotas(notas)
    st.write(descreverComparacao(notas))
    st.dataframe(tabelaDeNotas(notas), hide_index=True)
    st.caption(descreverImagem(resultados["nomeImagem"]))
    mostrarDiferencaEntreFluxos(resultados)
    st.markdown("**Parâmetros usados nesta comparação**")
    st.dataframe(tabelaDeParametros(resultados["parametros"]), hide_index=True)
    with st.container(border=True):
        st.markdown(LIMITACAO_DO_PROJETO)
    janela = avaliacao.JANELA_SSIM
    mostrarDoisQuadros(
        "O que esta seção responde",
        "A pergunta do projeto: o realce tradicional ou a restauração física de McGlamery, qual "
        "chega mais perto da referência? Os dois fluxos partem da mesma foto, e cada um segue "
        f"o seu caminho. {COMBINACAO} mostra o que acontece quando os dois caminhos são somados.",
        "Por que usar PSNR e SSIM",
        "O PSNR mede o erro pixel a pixel contra a referência, em decibéis, e quanto maior, mais "
        f"parecido. O SSIM compara brilho, contraste e estrutura em janelas de {janela}×{janela} "
        "pixels e vai até 1. Os dois juntos dão uma medida objetiva, já que o PSNR nem sempre "
        "concorda com o olho. As referências do UIEB não são fotos sem água: foram escolhidas por "
        "voluntários entre os resultados de 12 métodos de melhoria. Funcionam como um controle "
        "validado por pessoas, e a nota mede o quanto cada saída se aproxima dele.",
    )
    mostrarComoFunciona(
        "O PSNR parte do erro médio entre a saída e a referência, pixel a pixel, e o transforma "
        "em decibéis: quanto menor o erro, maior o PSNR. O SSIM compara brilho, contraste e "
        "estrutura em janelas pequenas, em cada canal, e tira a média. No código, as duas são "
        "funções prontas do scikit-image."
    )
    mostrarTrechoDeCodigo([avaliacao.calcularPsnr, avaliacao.calcularSsim], [])


def caminhoDaBusca(nome):
    return busca.PASTA_SAIDA / f"{nome}.csv"


def buscaGlobalDisponivel():
    return all(caminhoDaBusca(nome).exists() for nome in ARQUIVOS_DA_BUSCA)


@st.cache_data
def carregarBuscaGlobal():
    return {nome: pd.read_csv(caminhoDaBusca(nome)) for nome in ARQUIVOS_DA_BUSCA}


def melhoresValores(tabela, parametros):
    indice = tabela["ssim_media"].idxmax()

    return {nome: tabela.at[indice, nome] for nome in parametros}


def colunaDaSaida(chave, versao):
    return f"{chave}_{versao}"


def mediasGlobais(porImagem, versao):
    notas = {"Imagem crua": (porImagem["psnr_crua"].mean(), porImagem["ssim_crua"].mean())}
    for rotulo, chave in SAIDAS_DA_BUSCA.items():
        coluna = colunaDaSaida(chave, versao)
        notas[rotulo] = (porImagem[f"psnr_{coluna}"].mean(), porImagem[f"ssim_{coluna}"].mean())

    return notas


def fotosAFrente(porImagem, coluna, colunaDeComparacao):
    return int((porImagem[f"ssim_{coluna}"] > porImagem[f"ssim_{colunaDeComparacao}"]).sum())


def tabelaGlobalDosMetodos(porImagem):
    projeto = mediasGlobais(porImagem, "projeto")
    melhores = mediasGlobais(porImagem, "global")
    total = len(porImagem)

    return {
        "saída": list(SAIDAS_DA_BUSCA),
        "PSNR, valores do projeto": [round(projeto[rotulo][0], 2) for rotulo in SAIDAS_DA_BUSCA],
        "SSIM, valores do projeto": [round(projeto[rotulo][1], 3) for rotulo in SAIDAS_DA_BUSCA],
        "PSNR, melhores valores": [round(melhores[rotulo][0], 2) for rotulo in SAIDAS_DA_BUSCA],
        "SSIM, melhores valores": [round(melhores[rotulo][1], 3) for rotulo in SAIDAS_DA_BUSCA],
        "fotos em que superou a crua no SSIM": [
            f"{fotosAFrente(porImagem, colunaDaSaida(chave, 'global'), 'crua')} de {total}"
            for chave in SAIDAS_DA_BUSCA.values()
        ],
    }


def valoresTestados(nome, valores):
    return " | ".join(textoDoParametro(nome, valor) for valor in valores)


def tabelaDaBusca(valoresPorParametro, melhor, total):
    return {
        "parâmetro": [NOMES_DOS_PARAMETROS[nome] for nome in valoresPorParametro],
        "valores testados": [
            valoresTestados(nome, valores) for nome, valores in valoresPorParametro.items()
        ],
        "valor do projeto": [valorDoProjeto(nome) for nome in valoresPorParametro],
        f"melhor nas {total} fotos": [
            textoDoParametro(nome, melhor[nome]) for nome in valoresPorParametro
        ],
    }


def mascaraDosOutrosNoMelhor(tabela, melhor, nome):
    mascara = pd.Series(True, index=tabela.index)
    for outro, valor in melhor.items():
        if outro != nome:
            mascara &= np.isclose(tabela[outro], valor)

    return mascara


def ssimVariandoUmParametro(tabela, melhor, nome):
    linhas = tabela[mascaraDosOutrosNoMelhor(tabela, melhor, nome)].sort_values(nome)

    return linhas[nome].tolist(), linhas["ssim_media"].tolist()


def parametrosVariados(valoresPorParametro):
    return [nome for nome, valores in valoresPorParametro.items() if len(valores) > 1]


def efeitoDosParametros(tabela, melhor, valoresPorParametro):
    efeitos = {}
    for nome in parametrosVariados(valoresPorParametro):
        valores, notas = ssimVariandoUmParametro(tabela, melhor, nome)
        efeitos[nome] = max(notas) - min(notas)

    return efeitos


def escreverEfeitosNasBarras(eixo, valores, limite):
    afastamento = limite * AFASTAMENTO_DO_TEXTO
    for posicao, valor in enumerate(valores):
        texto = decimalBrasileiro(valor, 3)
        eixo.text(
            valor + afastamento,
            posicao,
            f"{texto} · maior efeito" if posicao == 0 else texto,
            va="center",
            fontsize=9,
            color=COR_DO_TEXTO,
            fontweight="bold" if posicao == 0 else "normal",
        )


def graficoDoEfeito(efeitos, titulo):
    ordenados = sorted(efeitos, key=efeitos.get, reverse=True)
    valores = [efeitos[nome] for nome in ordenados]
    limite = max(max(valores), 0.001) * FOLGA_DO_GRAFICO_DO_EFEITO
    altura = ALTURA_POR_BARRA * len(valores) + ALTURA_EXTRA_DO_GRAFICO
    figura = Figure(
        figsize=(LARGURA_DO_GRAFICO_DO_EFEITO, altura),
        layout="constrained",
        facecolor=COR_DA_SUPERFICIE,
    )
    eixo = figura.subplots()
    cores = [COR_DE_DESTAQUE] + [COR_SEM_DESTAQUE] * (len(valores) - 1)
    eixo.barh(range(len(valores)), valores, height=ALTURA_DA_BARRA_DE_NOTAS, color=cores)
    escreverEfeitosNasBarras(eixo, valores, limite)
    formatarEixoDasNotas(eixo, [NOMES_DOS_PARAMETROS[nome] for nome in ordenados], limite, [])
    figura.suptitle(titulo, x=POSICAO_DO_TITULO, ha="left", fontsize=10, color=COR_DO_TEXTO)

    return figura


def contagemDosMelhores(melhoresPorFoto, nome, valores):
    contagem = melhoresPorFoto[nome].value_counts()

    return [int(contagem.get(valor, 0)) for valor in valores]


def escreverContagensNasColunas(eixo, contagens, escolhidas):
    for posicao, (contagem, escolhida) in enumerate(zip(contagens, escolhidas)):
        eixo.text(
            posicao,
            contagem,
            inteiroBrasileiro(contagem),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COR_DO_TEXTO,
            fontweight="bold" if escolhida else "normal",
        )


def formatarPainelDeContagem(eixo, nome, valores, teto):
    rotulos = [textoDoParametro(nome, valor) for valor in valores]
    titulo = textwrap.fill(NOMES_DOS_PARAMETROS[nome], LARGURA_DO_TITULO_DO_PAINEL)
    eixo.set_facecolor(COR_DA_SUPERFICIE)
    eixo.set_xticks(range(len(valores)), rotulos, fontsize=8, color=COR_DO_TEXTO)
    eixo.set_ylim(0, teto)
    eixo.set_yticks([])
    eixo.spines[["top", "right", "left"]].set_visible(False)
    eixo.spines["bottom"].set_color(COR_SEM_DESTAQUE)
    eixo.tick_params(axis="x", length=0)
    eixo.set_title(titulo, fontsize=9, color=COR_DO_TEXTO)


def coresDosMetodos(parametros):
    metodos = list(dict.fromkeys(METODOS_DOS_PARAMETROS[nome] for nome in parametros))

    return dict(zip(metodos, CORES_DOS_METODOS))


def desenharPainelDeContagem(eixo, nome, valores, contagens, melhor, teto, cor):
    escolhidas = [bool(np.isclose(valor, melhor[nome])) for valor in valores]
    cores = [to_rgba(cor, 1 if escolhida else OPACIDADE_SEM_ESCOLHA) for escolhida in escolhidas]
    eixo.bar(range(len(valores)), contagens, width=LARGURA_DA_COLUNA, color=cores)
    escreverContagensNasColunas(eixo, contagens, escolhidas)
    formatarPainelDeContagem(eixo, nome, valores, teto)


def legendaDosMetodos(figura, cores):
    marcas = [Patch(facecolor=cor, label=metodo) for metodo, cor in cores.items()]
    figura.legend(
        handles=marcas, loc="outside lower center", ncols=len(marcas), frameon=False, fontsize=9
    )


def graficoDosMelhoresPorFoto(melhoresPorFoto, valoresPorParametro, melhor, titulo):
    variados = parametrosVariados(valoresPorParametro)
    cores = coresDosMetodos(variados)
    contagens = {
        nome: contagemDosMelhores(melhoresPorFoto, nome, valoresPorParametro[nome])
        for nome in variados
    }
    larguras = [len(valoresPorParametro[nome]) for nome in variados]
    teto = max(max(contagem) for contagem in contagens.values()) * FOLGA_DAS_COLUNAS
    figura = Figure(
        figsize=(
            LARGURA_POR_VALOR * sum(larguras) + MARGEM_DO_GRAFICO_DE_CONTAGEM,
            ALTURA_DO_GRAFICO_DE_CONTAGEM,
        ),
        layout="constrained",
        facecolor=COR_DA_SUPERFICIE,
    )
    eixos = figura.subplots(1, len(variados), gridspec_kw={"width_ratios": larguras})
    for eixo, nome in zip(eixos, variados):
        cor = cores[METODOS_DOS_PARAMETROS[nome]]
        valores = valoresPorParametro[nome]
        desenharPainelDeContagem(eixo, nome, valores, contagens[nome], melhor, teto, cor)
    legendaDosMetodos(figura, cores)
    figura.suptitle(titulo, x=POSICAO_DO_TITULO, ha="left", fontsize=10, color=COR_DO_TEXTO)

    return figura


def escreverNotasDaComparacao(eixo, notaDoDive, nota, casas, unidade, limite):
    afastamento = limite * AFASTAMENTO_DO_TEXTO
    eixo.text(
        notaDoDive + afastamento,
        0,
        f"{decimalBrasileiro(notaDoDive, casas)}{unidade}",
        va="center",
        fontsize=9,
        color=COR_DO_TEXTO,
    )
    eixo.text(
        nota - afastamento,
        0,
        f"{decimalBrasileiro(nota, casas)}{unidade}",
        ha="right",
        va="center",
        fontsize=9,
        color=BRANCO,
        fontweight="bold",
    )


def formatarEixoDaComparacao(eixo, limite):
    eixo.set_facecolor(COR_DA_SUPERFICIE)
    eixo.set_xlim(0, limite)
    eixo.set_ylim(*LIMITES_DA_BARRA_UNICA)
    eixo.set_yticks([])
    eixo.xaxis.set_major_formatter(FuncFormatter(rotuloDecimalDoEixo))
    eixo.grid(axis="x", color=COR_DA_GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    eixo.spines[["top", "right", "left"]].set_visible(False)
    eixo.spines["bottom"].set_color(COR_SEM_DESTAQUE)
    eixo.tick_params(axis="x", labelsize=8, colors=COR_DO_TEXTO_SECUNDARIO)


def graficoDaComparacaoComDive(notaDoDive, nota, rotulo, casas, unidade, limite, titulo):
    figura = Figure(
        figsize=(LARGURA_DO_GRAFICO_DE_NOTAS, ALTURA_DA_COMPARACAO),
        layout="constrained",
        facecolor=COR_DA_SUPERFICIE,
    )
    eixo = figura.subplots()
    eixo.barh(
        0,
        notaDoDive,
        height=ALTURA_DA_BARRA_DE_FORA,
        color=COR_DO_DIVE,
        label="dive+, aplicativo comercial",
    )
    eixo.barh(
        0,
        nota,
        height=ALTURA_DA_BARRA_DE_DENTRO,
        color=COR_DO_PROJETO,
        edgecolor=COR_DA_SUPERFICIE,
        linewidth=ESPESSURA_DO_CONTORNO,
        label=f"{rotulo}, melhores valores",
    )
    escreverNotasDaComparacao(eixo, notaDoDive, nota, casas, unidade, limite)
    formatarEixoDaComparacao(eixo, limite)
    figura.legend(loc="outside lower center", ncols=2, frameon=False, fontsize=8)
    figura.suptitle(titulo, x=POSICAO_DO_TITULO, ha="left", fontsize=10, color=COR_DO_TEXTO)

    return figura


def melhorSaidaDoProjeto(porImagem):
    melhores = mediasGlobais(porImagem, "global")
    ssims = {rotulo: melhores[rotulo][1] for rotulo in SAIDAS_DA_BUSCA}

    return max(ssims, key=ssims.get)


def porcentagemDoDive(nota, notaDoDive):
    return round(100 * nota / notaDoDive)


def subidaDoSsim(porImagem, chave):
    globalMedio = decimalBrasileiro(porImagem[f"ssim_{chave}_global"].mean(), 3)
    porFotoMedio = decimalBrasileiro(porImagem[f"ssim_{chave}_porFoto"].mean(), 3)

    return f"de {globalMedio} para {porFotoMedio}"


def textoDaMelhorDeCadaFoto(porImagem):
    return (
        "Se cada foto usasse a sua própria melhor combinação, o SSIM médio do Fluxo 1 subiria "
        f"{subidaDoSsim(porImagem, 'fluxo1')}, e o do Fluxo 2 "
        f"{subidaDoSsim(porImagem, 'fluxo2')}."
    )


def escolhaDiferenteDaMaisFrequente(melhoresPorFoto, valoresPorParametro, melhor):
    for nome in parametrosVariados(valoresPorParametro):
        valores = valoresPorParametro[nome]
        contagens = contagemDosMelhores(melhoresPorFoto, nome, valores)
        if not np.isclose(valores[int(np.argmax(contagens))], melhor[nome]):
            return True

    return False


def comMinuscula(texto):
    return texto[:1].lower() + texto[1:]


def parametrosNaPonta(valoresPorParametro, melhor):
    return [
        nome
        for nome, valores in valoresPorParametro.items()
        if len(valores) >= VALORES_PARA_TER_MEIO and melhor[nome] in (min(valores), max(valores))
    ]


def textoDosParametrosNaPonta(valoresPorParametro, melhor):
    naPonta = parametrosNaPonta(valoresPorParametro, melhor)
    if not naPonta:
        return "Os melhores valores ficaram no meio da lista testada."
    itens = ", ".join(
        f"{comMinuscula(NOMES_DOS_PARAMETROS[nome])} em {textoDoParametro(nome, melhor[nome])}"
        for nome in naPonta
    )

    return (
        f"Ficaram na ponta da lista testada: {itens}. Nesses casos, pode haver um valor um pouco "
        "melhor fora da lista, se o parâmetro permitir."
    )


def mostrarComoOUiebAvalia():
    st.markdown("**Como o UIEB chega aos resultados dele**")
    st.markdown(COMO_O_UIEB_AVALIA)
    esquerda, direita = st.columns(2)
    esquerda.pyplot(graficoDoPsnr(NOTAS_DO_ARTIGO_DO_UIEB))
    direita.pyplot(graficoDoSsim(NOTAS_DO_ARTIGO_DO_UIEB))
    st.caption(
        "Médias nas 890 fotos, na tabela II do artigo do UIEB (Li et al., 2019). Red Channel, "
        "GDCP e UDCP usam modelo físico, e o UDCP usa o canal escuro com verde e azul, a mesma "
        "ideia da etapa 1 do Fluxo 2. O dive+ e a fusão deram origem a 68% das referências, o que "
        "lhes dá vantagem. Como o cálculo do SSIM pode variar entre implementações, a comparação "
        "com o projeto é aproximada."
    )


def mostrarComoChegamos(dados, total):
    combinacoesDaExploracao = len(busca.combinacoes(busca.VALORES_DA_EXPLORACAO))
    st.markdown("**Como chegamos ao resultado geral**")
    st.write(
        f"Do mesmo jeito que o UIEB, a nota é a média do SSIM nas {total} fotos. Para cada fluxo, "
        "montamos uma lista de valores para cada parâmetro, sempre com o valor do projeto, e "
        "testamos todas as combinações desses valores em todas as fotos. A melhor combinação é "
        "a de maior SSIM médio. As listas do Fluxo 1 saíram de uma exploração com "
        f"{combinacoesDaExploracao} combinações em {busca.FOTOS_DA_EXPLORACAO} fotos."
    )
    abas = st.tabs([FLUXO_TRADICIONAL, FLUXO_MCGLAMERY, COMBINACAO])
    buscas = [
        (busca.VALORES_DO_FLUXO_1, dados["fluxo1"]),
        (busca.VALORES_DO_FLUXO_2, dados["fluxo2"]),
        (busca.VALORES_DA_COMBINACAO, dados["combinacao"]),
    ]
    for aba, (valores, tabela) in zip(abas, buscas):
        with aba:
            st.caption(f"{len(tabela)} combinações testadas em cada uma das {total} fotos")
            melhor = melhoresValores(tabela, valores)
            st.dataframe(tabelaDaBusca(valores, melhor, total), hide_index=True)
    st.caption(
        f"Em {COMBINACAO}, a parte tradicional usa os melhores valores do Fluxo 1, a janela e o "
        "raio vêm do melhor Fluxo 2, e só o piso é testado."
    )


def mostrarEfeitoDosParametros(dados):
    st.markdown("**Qual parâmetro mais pesa**")
    st.write(
        "Partindo da melhor combinação, cada parâmetro foi trocado pelos outros valores testados. "
        "A barra mostra quanto o SSIM médio muda entre o pior e o melhor valor."
    )
    buscas = [
        (FLUXO_TRADICIONAL, busca.VALORES_DO_FLUXO_1, dados["fluxo1"]),
        (FLUXO_MCGLAMERY, busca.VALORES_DO_FLUXO_2, dados["fluxo2"]),
    ]
    colunas = st.columns(2)
    for coluna, (rotulo, valores, tabela) in zip(colunas, buscas):
        melhor = melhoresValores(tabela, valores)
        efeitos = efeitoDosParametros(tabela, melhor, valores)
        coluna.pyplot(graficoDoEfeito(efeitos, f"{rotulo}: quanto o SSIM médio muda"))
        coluna.caption(textoDosParametrosNaPonta(valores, melhor))


def mostrarComportamentoDosMetodos(porImagem):
    total = len(porImagem)
    st.markdown(f"**Como cada método se comporta nas {total} fotos**")
    melhores = mediasGlobais(porImagem, "global")
    esquerda, direita = st.columns(2)
    esquerda.pyplot(graficoDoPsnr(melhores))
    direita.pyplot(graficoDoSsim(melhores))
    st.caption(
        f"Médias nas {total} fotos, com os melhores valores de cada fluxo. Em azul, a maior nota "
        "de cada gráfico."
    )
    st.dataframe(tabelaGlobalDosMetodos(porImagem), hide_index=True)
    psnrCrua, ssimCrua = melhores["Imagem crua"]
    vitorias = fotosAFrente(porImagem, "fluxo2_global", "fluxo1_global")
    st.write(
        f"A foto crua, sem nenhum tratamento, tem PSNR médio {decimalBrasileiro(psnrCrua, 2)} dB "
        f"e SSIM médio {decimalBrasileiro(ssimCrua, 3)}. Com os melhores valores de cada um, o "
        f"Fluxo 2 superou o Fluxo 1 no SSIM em {vitorias} de {total} fotos."
    )


def mostrarMelhoresPorFoto(dados):
    total = len(dados["por_imagem"])
    st.markdown("**Em quantas fotos cada valor foi o melhor**")
    st.write(
        "Para cada foto, achamos a combinação que deu o maior SSIM naquela foto. Cada coluna "
        f"conta em quantas das {total} fotos aquele valor fazia parte dessa combinação. Cada "
        "método tem a sua cor, e a coluna de cor mais forte é o valor escolhido para a base "
        f"inteira. {textoDaMelhorDeCadaFoto(dados['por_imagem'])}"
    )
    buscas = [
        (FLUXO_TRADICIONAL, busca.VALORES_DO_FLUXO_1, dados["fluxo1"]),
        (FLUXO_MCGLAMERY, busca.VALORES_DO_FLUXO_2, dados["fluxo2"]),
    ]
    diferentes = []
    abas = st.tabs([rotulo for rotulo, valores, tabela in buscas])
    for aba, (rotulo, valores, tabela) in zip(abas, buscas):
        melhor = melhoresValores(tabela, valores)
        aba.pyplot(graficoDosMelhoresPorFoto(dados["melhores_por_foto"], valores, melhor, rotulo))
        diferentes.append(
            escolhaDiferenteDaMaisFrequente(dados["melhores_por_foto"], valores, melhor)
        )
    if any(diferentes):
        st.caption(
            "O valor escolhido nem sempre é o que mais vence foto a foto. Ele é o que dá a maior "
            "média, somando o que se ganha e o que se perde em todas as fotos."
        )


def mostrarComparacaoComDive(porImagem):
    rotulo = melhorSaidaDoProjeto(porImagem)
    psnr, ssim = mediasGlobais(porImagem, "global")[rotulo]
    psnrDoDive, ssimDoDive = NOTAS_DO_ARTIGO_DO_UIEB["dive+"]
    st.markdown("**O quão perto chegamos do dive+**")
    st.write(
        "A barra azul é o dive+, o aplicativo comercial que ficou em primeiro na tabela do UIEB. "
        f"A vermelha, por dentro, é o {rotulo} com os melhores valores, a nossa melhor saída. "
        "Quanto mais a vermelha preenche a azul, mais perto chegamos."
    )
    esquerda, direita = st.columns(2)
    esquerda.pyplot(
        graficoDaComparacaoComDive(
            psnrDoDive,
            psnr,
            rotulo,
            2,
            " dB",
            psnrDoDive * FOLGA_DA_COMPARACAO_DO_PSNR,
            f"PSNR (dB): {porcentagemDoDive(psnr, psnrDoDive)}% do dive+",
        )
    )
    direita.pyplot(
        graficoDaComparacaoComDive(
            ssimDoDive,
            ssim,
            rotulo,
            3,
            "",
            LIMITE_DA_COMPARACAO_DO_SSIM,
            f"SSIM: {porcentagemDoDive(ssim, ssimDoDive)}% do dive+",
        )
    )
    st.caption(
        "O dive+ deu origem a 43,9% das referências do UIEB, então tem vantagem nessa comparação. "
        "As notas dele vêm do artigo, e o SSIM pode variar um pouco entre implementações."
    )


def paginaResultadosGlobais():
    st.header(RESULTADOS_GLOBAIS)
    st.caption("As 890 fotos do UIEB, com a melhor combinação de parâmetros de cada fluxo")
    mostrarComoOUiebAvalia()
    if not buscaGlobalDisponivel():
        st.warning(
            "Os resultados da busca ainda não foram gerados. Coloque a base completa em dados/ e "
            "rode python executar_busca_global.py."
        )
        return
    dados = carregarBuscaGlobal()
    total = len(dados["por_imagem"])
    mostrarComoChegamos(dados, total)
    mostrarEfeitoDosParametros(dados)
    mostrarComportamentoDosMetodos(dados["por_imagem"])
    mostrarMelhoresPorFoto(dados)
    mostrarComparacaoComDive(dados["por_imagem"])
    with st.expander("Ver o código da busca"):
        for funcao in [
            busca.combinacoes,
            busca.notasDoFluxo1,
            busca.cenasRestauradas,
            busca.indiceDaMelhor,
        ]:
            mostrarFuncao(funcao)


@st.cache_data(show_spinner=False)
def quantidadeDeSlides(caminho, modificado):
    with pymupdf.open(caminho) as documento:
        return documento.page_count


@st.cache_data(show_spinner="Preparando o slide...")
def imagemDoSlide(caminho, modificado, numero):
    with pymupdf.open(caminho) as documento:
        return documento[numero - 1].get_pixmap(dpi=RESOLUCAO_DOS_SLIDES).tobytes("png")


def mudarDeSlide(passo, total):
    st.session_state[CHAVE_DO_SLIDE] = min(max(st.session_state[CHAVE_DO_SLIDE] + passo, 1), total)


def mostrarControlesDosSlides(total):
    atual = st.session_state[CHAVE_DO_SLIDE]
    anterior, controle, proximo = st.columns([1, 6, 1], vertical_alignment="center")
    anterior.button(
        "◀ Anterior",
        key="slideAnterior",
        on_click=mudarDeSlide,
        args=(-1, total),
        disabled=atual == 1,
    )
    controle.slider("Slide", 1, total, key=CHAVE_DO_SLIDE, label_visibility="collapsed")
    proximo.button(
        "Próximo ▶",
        key="slideProximo",
        on_click=mudarDeSlide,
        args=(1, total),
        disabled=atual == total,
    )


def mostrarBotoesDoPdf():
    esquerda, direita = st.columns(2)
    esquerda.link_button("Abrir em tela cheia, no visualizador do navegador", ENDERECO_DOS_SLIDES)
    direita.download_button(
        "Baixar o PDF",
        ARQUIVO_DOS_SLIDES.read_bytes(),
        file_name=ARQUIVO_DOS_SLIDES.name,
        mime="application/pdf",
    )
    st.caption(
        "Na nova aba, F11 deixa a apresentação em tela cheia, e as setas do teclado passam os "
        "slides. Aqui no app, passe o mouse sobre o slide e use o botão de expandir para vê-lo "
        "maior."
    )


def abrirSlides():
    st.session_state[CHAVE_DO_MODO_SLIDES] = True


def fecharSlides():
    st.session_state[CHAVE_DO_MODO_SLIDES] = False


def mostrarGoogleSlides():
    st.iframe(GOOGLE_SLIDES_EMBUTIDO, height=ALTURA_DO_GOOGLE_SLIDES)
    st.link_button("Abrir no Google Slides, em modo apresentação", GOOGLE_SLIDES_APRESENTACAO)
    st.caption(
        "Clique no slide e use as setas do teclado para passar. O botão de tela cheia fica na "
        "barra de baixo da apresentação."
    )


def paginaDosSlides():
    titulo, voltar = st.columns([5, 1], vertical_alignment="center")
    titulo.header(SLIDES)
    voltar.button("Voltar ao app", key="voltarDosSlides", on_click=fecharSlides)
    abaGoogle, abaPdf = st.tabs(["Google Slides", "PDF"])
    with abaGoogle:
        mostrarGoogleSlides()
    with abaPdf:
        mostrarSlidesDoPdf()


def mostrarSlidesDoPdf():
    if not ARQUIVO_DOS_SLIDES.is_file():
        st.info(
            "Coloque o PDF dos slides em static/slides.pdf, dentro da pasta do projeto, e "
            "recarregue a página."
        )
        return
    caminho = str(ARQUIVO_DOS_SLIDES)
    modificado = ARQUIVO_DOS_SLIDES.stat().st_mtime
    total = quantidadeDeSlides(caminho, modificado)
    st.session_state[CHAVE_DO_SLIDE] = min(st.session_state.get(CHAVE_DO_SLIDE, 1), total)
    mostrarControlesDosSlides(total)
    numero = st.session_state[CHAVE_DO_SLIDE]
    st.image(imagemDoSlide(caminho, modificado, numero), caption=f"Slide {numero} de {total}")
    mostrarBotoesDoPdf()


def mostrarNavegacao():
    st.divider()
    posicao = posicaoAtual()
    colunaAnterior, espaco, colunaProxima = st.columns([1, 4, 1])
    colunaAnterior.button("Anterior", on_click=mover, args=(-1,), disabled=posicao == 0)
    colunaProxima.button(
        "Próxima", on_click=mover, args=(1,), disabled=posicao == len(ordemLinear()) - 1
    )


def chaveDaEtapa(secao):
    return f"etapa::{secao}"


def ordemLinear():
    paradasDosFluxos = [
        (fluxo, etapa) for fluxo in ETAPAS_DO_FLUXO for etapa in ETAPAS_DO_FLUXO[fluxo]
    ]

    paradasFinais = [(RESULTADO, None), (RESULTADOS_GLOBAIS, None)]

    return [(INTRODUCAO, None)] + paradasDosFluxos + paradasFinais


def posicaoAtual():
    secao = st.session_state.secao

    return ordemLinear().index((secao, st.session_state.get(chaveDaEtapa(secao))))


def mover(passo):
    ordem = ordemLinear()
    indice = min(max(posicaoAtual() + passo, 0), len(ordem) - 1)
    secao, etapa = ordem[indice]
    st.session_state.secao = secao
    if etapa is not None:
        st.session_state[chaveDaEtapa(secao)] = etapa


# ================================================================================================

st.set_page_config(page_title="McGlammery", layout="wide")

if not (PASTA_BASE / base_de_dados.PASTA_CRUAS).is_dir():
    st.error(
        "Base não encontrada. Coloque as pastas raw-890 e reference-890 da base UIEB dentro de "
        "dados/, ou use a amostra que vem com o projeto, na pasta amostra/."
    )
    st.stop()

nomesDasImagens = base_de_dados.listarNomesDasImagens(PASTA_BASE / base_de_dados.PASTA_CRUAS)

modoSlides = st.session_state.get(CHAVE_DO_MODO_SLIDES, False)

st.sidebar.title("McGlammery")
if modoSlides:
    st.sidebar.button("Voltar ao app", key="fecharSlides", on_click=fecharSlides, type="primary")
else:
    st.sidebar.button("Abrir slides", key="abrirSlides", on_click=abrirSlides, type="primary")
st.sidebar.link_button("Abrir no Google Slides", GOOGLE_SLIDES_APRESENTACAO)
st.sidebar.link_button("Ver o código-fonte no GitHub", CODIGO_FONTE)
nomeImagem = st.sidebar.selectbox(
    "Imagem do raw-890", nomesDasImagens, index=nomesDasImagens.index(IMAGEM_PADRAO)
)
st.sidebar.caption(descreverImagem(nomeImagem))
if PASTA_BASE.name == base_de_dados.PASTA_AMOSTRA:
    st.sidebar.caption(
        f"Usando a amostra do projeto, com {len(nomesDasImagens)} fotos. Para ver as 890, coloque "
        "a base UIEB completa em dados/."
    )
secao = st.sidebar.selectbox("Seção", SECOES, key="secao", on_change=fecharSlides)
etapa = None
if secao in ETAPAS_DO_FLUXO:
    etapa = st.sidebar.radio("Etapa do fluxo", ETAPAS_DO_FLUXO[secao], key=chaveDaEtapa(secao))
    st.sidebar.caption(GUIAS_DOS_FLUXOS[secao])

parametros = parametrosAtuais()
alterados = parametrosAlterados(parametros)
if alterados:
    st.sidebar.warning(
        "Parâmetros diferentes dos do projeto: "
        + ", ".join(NOMES_DOS_PARAMETROS[nome] for nome in alterados)
        + "."
    )
    st.sidebar.button("Voltar aos valores do projeto", on_click=voltarAosValoresDoProjeto)

if modoSlides:
    paginaDosSlides()
    st.stop()

resultados = processarImagem(nomeImagem, parametros)

if alterados:
    st.info(
        "Alguns parâmetros estão diferentes dos do projeto. As imagens e as notas usam os valores "
        "escolhidos, e os textos explicam os valores do projeto."
    )

if secao == INTRODUCAO:
    paginaIntroducao(resultados)
elif secao == RESULTADO:
    paginaResultado(resultados)
elif secao == RESULTADOS_GLOBAIS:
    paginaResultadosGlobais()
elif etapa == MODELO:
    paginaModelo(secao, resultados)
else:
    mostrarEtapa(secao, etapa, resultados)

mostrarNavegacao()
