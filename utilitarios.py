from pathlib import Path

import cv2
import numpy as np

INTENSIDADE_MAXIMA = 255


def lerImagem(nomeImagem, pasta, modoLeitura):
    bytesDoArquivo = np.fromfile(Path(pasta) / nomeImagem, dtype=np.uint8)
    imagem = cv2.imdecode(bytesDoArquivo, modoLeitura)

    return imagem


def salvarImagem(imagem, nome, pastaSaida):
    pastaSaida = Path(pastaSaida)
    pastaSaida.mkdir(parents=True, exist_ok=True)

    caminho = pastaSaida / (nome + ".png")

    ok, bytesCodificados = cv2.imencode(".png", imagem)

    if not ok:
        raise ValueError(f"não foi possível codificar a imagem: {caminho}")

    bytesCodificados.tofile(caminho)

    return caminho


def limitarEntreZeroEUm(imagem):
    return np.clip(imagem, 0, 1)


def paraNiveisDeCinza(imagem):
    return np.round(imagem * INTENSIDADE_MAXIMA).astype(np.uint8)


def rgbParaBgr(imagemRgb):
    return imagemRgb[:, :, ::-1]


def salvarImagemNormalizada(imagem, nome, pastaSaida):
    niveis = paraNiveisDeCinza(limitarEntreZeroEUm(imagem))

    if niveis.ndim == 3:
        niveis = rgbParaBgr(niveis)

    return salvarImagem(niveis, nome, pastaSaida)
