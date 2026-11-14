import cv2
import numpy as np

from utilitarios import INTENSIDADE_MAXIMA, limitarEntreZeroEUm, paraNiveisDeCinza

LUMINOSIDADE = 0
GANHO_MAXIMO = 1.3
LIMITE_CONTRASTE_CLAHE = 2.0
BLOCOS_CLAHE = 8
GANHO_BAIXAS_FREQUENCIAS = 0.8
GANHO_ALTAS_FREQUENCIAS = 1.2
FREQUENCIA_DE_CORTE = 30
NITIDEZ_DA_TRANSICAO = 1


# ================================= Etapa 4 - Cor (Mundo Cinza) ==================================


def mediasDosCanais(imagem):
    return imagem.mean(axis=(0, 1))


def limitarGanhos(ganhos, ganhoMaximo=GANHO_MAXIMO):
    return np.minimum(ganhos, ganhoMaximo)


def ganhosDoMundoCinza(imagem, ganhoMaximo=GANHO_MAXIMO):
    medias = mediasDosCanais(imagem)
    ganhos = medias.mean() / medias

    return limitarGanhos(ganhos, ganhoMaximo)


def aplicarMundoCinza(imagem, ganhoMaximo=GANHO_MAXIMO):
    return limitarEntreZeroEUm(imagem * ganhosDoMundoCinza(imagem, ganhoMaximo))


# ================================= Etapa 5 - Contraste (CLAHE) ==================================


def rgbParaLab(imagem):
    return cv2.cvtColor(paraNiveisDeCinza(limitarEntreZeroEUm(imagem)), cv2.COLOR_RGB2LAB)


def labParaRgb(lab):
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float64) / INTENSIDADE_MAXIMA


def trocarLuminosidade(lab, luminosidade):
    atualizada = lab.copy()
    atualizada[:, :, LUMINOSIDADE] = luminosidade

    return atualizada


def equalizarLuminosidade(luminosidade, limite=LIMITE_CONTRASTE_CLAHE, blocos=BLOCOS_CLAHE):
    clahe = cv2.createCLAHE(clipLimit=limite, tileGridSize=(blocos, blocos))

    return clahe.apply(luminosidade)


def aplicarClahe(imagem, limite=LIMITE_CONTRASTE_CLAHE, blocos=BLOCOS_CLAHE):
    lab = rgbParaLab(imagem)
    equalizada = equalizarLuminosidade(lab[:, :, LUMINOSIDADE], limite, blocos)

    return labParaRgb(trocarLuminosidade(lab, equalizada))


# ========================= Etapa 6 - Iluminação (Filtragem Homomórfica) =========================


def paraDominioLogaritmico(luminosidade):
    return np.log1p(luminosidade.astype(np.float64))


def espectroCentralizado(canal):
    return np.fft.fftshift(np.fft.fft2(canal))


def distanciaAoCentroAoQuadrado(altura, largura):
    linhas = np.arange(altura) - altura // 2
    colunas = np.arange(largura) - largura // 2

    return linhas[:, np.newaxis] ** 2 + colunas[np.newaxis, :] ** 2


def filtroEnfaseAltasFrequencias(
    altura,
    largura,
    ganhoBaixas=GANHO_BAIXAS_FREQUENCIAS,
    ganhoAltas=GANHO_ALTAS_FREQUENCIAS,
    corte=FREQUENCIA_DE_CORTE,
):
    distanciaAoQuadrado = distanciaAoCentroAoQuadrado(altura, largura)
    passaAltas = 1 - np.exp(-NITIDEZ_DA_TRANSICAO * distanciaAoQuadrado / corte**2)
    amplitude = ganhoAltas - ganhoBaixas
    ganhos = amplitude * passaAltas + ganhoBaixas
    ganhos[altura // 2, largura // 2] = 1

    return ganhos


def voltarAoDominioEspacial(espectro):
    return np.real(np.fft.ifft2(np.fft.ifftshift(espectro)))


def sairDoDominioLogaritmico(canal):
    return np.expm1(canal)


def paraNiveisDeLuminosidade(canal):
    return np.round(np.clip(canal, 0, INTENSIDADE_MAXIMA)).astype(np.uint8)


def filtrarLuminosidade(
    luminosidade,
    ganhoBaixas=GANHO_BAIXAS_FREQUENCIAS,
    ganhoAltas=GANHO_ALTAS_FREQUENCIAS,
    corte=FREQUENCIA_DE_CORTE,
):
    espectro = espectroCentralizado(paraDominioLogaritmico(luminosidade))
    filtro = filtroEnfaseAltasFrequencias(*luminosidade.shape, ganhoBaixas, ganhoAltas, corte)
    filtrada = voltarAoDominioEspacial(espectro * filtro)

    return paraNiveisDeLuminosidade(sairDoDominioLogaritmico(filtrada))


def aplicarFiltragemHomomorfica(
    imagem,
    ganhoBaixas=GANHO_BAIXAS_FREQUENCIAS,
    ganhoAltas=GANHO_ALTAS_FREQUENCIAS,
    corte=FREQUENCIA_DE_CORTE,
):
    lab = rgbParaLab(imagem)
    filtrada = filtrarLuminosidade(lab[:, :, LUMINOSIDADE], ganhoBaixas, ganhoAltas, corte)

    return labParaRgb(trocarLuminosidade(lab, filtrada))
