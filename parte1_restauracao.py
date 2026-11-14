import cv2
import numpy as np

from utilitarios import limitarEntreZeroEUm

VERDE = 1
AZUL = 2
JANELA_CANAL_ESCURO = 15
FRACAO_REGIAO_DE_BUSCA = 1 / 3
LIMITE_QUASE_BRANCO = 60 / 255
FRACAO_MAIS_CLAROS = 0.001
RAIO_FILTRO_GUIADO = 30
EPSILON_FILTRO_GUIADO = 1e-3
TRANSMISSAO_MINIMA = 0.5


# ============================ Etapa 1 - Canal Escuro (Dark Channel) =============================


def minimoEntreVerdeEAzul(imagem):
    return np.minimum(imagem[:, :, VERDE], imagem[:, :, AZUL])


def erosaoEmTonsDeCinza(canal, tamanhoJanela):
    elementoEstruturante = np.ones((tamanhoJanela, tamanhoJanela), dtype=np.uint8)

    return cv2.erode(canal, elementoEstruturante)


def calcularCanalEscuro(imagem, janela=JANELA_CANAL_ESCURO):
    return erosaoEmTonsDeCinza(minimoEntreVerdeEAzul(imagem), janela)


# =================================== Etapa 2 - Cor da Água B ====================================


def mascaraTercoSuperior(altura, largura):
    mascara = np.zeros((altura, largura), dtype=bool)
    mascara[: int(altura * FRACAO_REGIAO_DE_BUSCA)] = True

    return mascara


def diferencaEntreCanais(imagem):
    return imagem.max(axis=2) - imagem.min(axis=2)


def mascaraQuaseBranco(imagem):
    return diferencaEntreCanais(imagem) <= LIMITE_QUASE_BRANCO


def mascaraDeCandidatos(imagem):
    tercoSuperior = mascaraTercoSuperior(*imagem.shape[:2])
    candidatos = tercoSuperior & ~mascaraQuaseBranco(imagem)

    return candidatos if candidatos.any() else tercoSuperior


def quantidadeMaisClaros(total):
    return max(1, round(total * FRACAO_MAIS_CLAROS))


def posicoesMaisClaras(canalEscuro, candidatos):
    linhas, colunas = np.nonzero(candidatos)
    valores = canalEscuro[linhas, colunas]
    maisClaros = np.argsort(valores)[-quantidadeMaisClaros(valores.size):]

    return linhas[maisClaros], colunas[maisClaros]


def calcularIntensidade(imagem):
    return imagem.mean(axis=2)


def posicaoMaisIntensa(imagem, linhas, colunas):
    melhor = np.argmax(calcularIntensidade(imagem)[linhas, colunas])

    return linhas[melhor], colunas[melhor]


def estimarCorDaAgua(imagem, canalEscuro):
    candidatos = mascaraDeCandidatos(imagem)
    linhas, colunas = posicoesMaisClaras(canalEscuro, candidatos)
    linha, coluna = posicaoMaisIntensa(imagem, linhas, colunas)

    return imagem[linha, coluna], (linha, coluna)


# =================================== Etapa 3a - Normalização ====================================


def divisaoSegura(numerador, denominador):
    return np.divide(
        numerador, denominador, out=np.zeros_like(numerador), where=denominador > 0
    )


def normalizarPelaCorDaAgua(imagem, corDaAgua):
    return divisaoSegura(imagem, corDaAgua)


# ==================================== Etapa 3b - Transmissão ====================================


def calcularTransmissao(normalizada, janela=JANELA_CANAL_ESCURO):
    return limitarEntreZeroEUm(1 - calcularCanalEscuro(normalizada, janela))


# ==================================== Etapa 3c - Refinamento ====================================


def filtroGuiado(guia, alvo, raio, epsilon):
    filtrada = cv2.ximgproc.guidedFilter(
        guia.astype(np.float32), alvo.astype(np.float32), raio, epsilon
    )

    return filtrada.astype(np.float64)


def refinarTransmissao(transmissao, imagem, raio=RAIO_FILTRO_GUIADO):
    guia = calcularIntensidade(imagem)
    refinada = filtroGuiado(guia, transmissao, raio, EPSILON_FILTRO_GUIADO)

    return limitarEntreZeroEUm(refinada)


# ================================= Etapa 3d - Recuperação de J ==================================


def limitarTransmissao(transmissao, piso=TRANSMISSAO_MINIMA):
    return np.maximum(transmissao, piso)


def adicionarEixoDeCanal(canal):
    return canal[:, :, np.newaxis]


def recuperarCena(imagem, corDaAgua, transmissao, piso=TRANSMISSAO_MINIMA):
    limitada = adicionarEixoDeCanal(limitarTransmissao(transmissao, piso))
    cena = (imagem - corDaAgua) / limitada + corDaAgua

    return limitarEntreZeroEUm(cena)
