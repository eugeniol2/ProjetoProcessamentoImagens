import cv2
import numpy as np

from utilitarios import INTENSIDADE_MAXIMA, lerImagem

PASTA_DADOS = "dados"
PASTA_AMOSTRA = "amostra"
PASTA_CRUAS = "raw-890"
PASTA_REFERENCIAS = "reference-890"
PASTA_DESAFIADORAS = "challenging-60"
EXTENSAO_IMAGEM = ".png"


def pastaDaBase(pastaProjeto):
    completa = pastaProjeto / PASTA_DADOS

    return completa if (completa / PASTA_CRUAS).is_dir() else pastaProjeto / PASTA_AMOSTRA


def listarNomesDasImagens(pasta):
    if not pasta.is_dir():
        return []

    return sorted(caminho.name for caminho in pasta.iterdir() if caminho.suffix == EXTENSAO_IMAGEM)


def bgrParaRgb(imagemBgr):
    return imagemBgr[:, :, ::-1]


def normalizarEntreZeroEUm(imagem):
    return imagem.astype(np.float64) / INTENSIDADE_MAXIMA


def carregarImagem(nomeImagem, pasta):
    imagemBgr = lerImagem(nomeImagem, pasta, cv2.IMREAD_COLOR)
    imagemRgb = bgrParaRgb(imagemBgr)

    return normalizarEntreZeroEUm(imagemRgb)


def carregarPar(nomeImagem, pastaBase):
    crua = carregarImagem(nomeImagem, pastaBase / PASTA_CRUAS)
    referencia = carregarImagem(nomeImagem, pastaBase / PASTA_REFERENCIAS)

    return crua, referencia
