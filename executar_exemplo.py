import sys
from pathlib import Path

import cv2
import numpy as np

from avaliacao import calcularPsnr, calcularSsim
from base_de_dados import (
    PASTA_CRUAS,
    PASTA_DESAFIADORAS,
    carregarPar,
    listarNomesDasImagens,
    pastaDaBase,
)
from parte1_restauracao import (
    TRANSMISSAO_MINIMA,
    calcularCanalEscuro,
    calcularTransmissao,
    estimarCorDaAgua,
    normalizarPelaCorDaAgua,
    recuperarCena,
    refinarTransmissao,
)
from parte2_tradicional import (
    aplicarClahe,
    aplicarFiltragemHomomorfica,
    aplicarMundoCinza,
    ganhosDoMundoCinza,
)
from utilitarios import salvarImagemNormalizada

PASTA_PROJETO = Path(__file__).parent
PASTA_BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else pastaDaBase(PASTA_PROJETO)
PASTA_SAIDA = PASTA_PROJETO / "resultados" / "exemplo"
IMAGEM_EXEMPLO = "13_img_.png"
RAIO_MARCA = 20
ESPESSURA_MARCA = 3
VERMELHO = (1.0, 0.0, 0.0)


def imprimirBase(nomesCruas, nomesDesafiadoras):
    print(f"raw-890: {len(nomesCruas)} imagens | challenging-60: {len(nomesDesafiadoras)} imagens")


def imprimirResumoDaImagem(rotulo, imagem):
    altura, largura = imagem.shape[:2]
    vermelho, verde, azul = imagem.mean(axis=(0, 1))
    print(f"{rotulo}: {largura}x{altura} | médias R {vermelho:.2f} G {verde:.2f} B {azul:.2f}")


def imprimirCorDaAgua(corDaAgua, posicao):
    vermelho, verde, azul = corDaAgua
    linha, coluna = posicao
    print(
        f"cor da água B: R {vermelho:.2f} G {verde:.2f} B {azul:.2f} | "
        f"linha {linha}, coluna {coluna}"
    )


def imprimirResumoDaTransmissao(rotulo, transmissao):
    abaixoDoLimite = np.mean(transmissao < TRANSMISSAO_MINIMA) * 100
    print(
        f"{rotulo}: mín {transmissao.min():.2f} | média {transmissao.mean():.2f} | "
        f"máx {transmissao.max():.2f} | abaixo de t0: {abaixoDoLimite:.0f}%"
    )


def imprimirGanhos(rotulo, ganhos):
    vermelho, verde, azul = ganhos
    print(f"ganhos do mundo cinza ({rotulo}): R {vermelho:.2f} G {verde:.2f} B {azul:.2f}")


def imprimirNotas(rotulo, imagem, referencia):
    psnr = calcularPsnr(imagem, referencia)
    ssim = calcularSsim(imagem, referencia)
    print(f"{rotulo}: PSNR {psnr:.2f} dB | SSIM {ssim:.3f}")


def cinzaParaRgb(canal):
    return np.stack([canal, canal, canal], axis=-1)


def marcarPosicao(canal, posicao):
    marcada = cinzaParaRgb(canal)
    linha, coluna = posicao
    cv2.circle(marcada, (int(coluna), int(linha)), RAIO_MARCA, VERMELHO, ESPESSURA_MARCA)

    return marcada


# ================================================================================================

# ======================================== Base de Dados =========================================

nomesCruas = listarNomesDasImagens(PASTA_BASE / PASTA_CRUAS)
nomesDesafiadoras = listarNomesDasImagens(PASTA_BASE / PASTA_DESAFIADORAS)

print(f"base usada: {PASTA_BASE}")
imprimirBase(nomesCruas, nomesDesafiadoras)

crua, referencia = carregarPar(IMAGEM_EXEMPLO, PASTA_BASE)

print(f"{IMAGEM_EXEMPLO}: imagem piloto do projeto")
imprimirResumoDaImagem("original", crua)
imprimirResumoDaImagem("referência", referencia)

salvarImagemNormalizada(crua, "0_original", PASTA_SAIDA)
salvarImagemNormalizada(referencia, "referencia", PASTA_SAIDA)


# ========================= Parte 1 - Variáveis do Modelo e Restauração ==========================

canalEscuro = calcularCanalEscuro(crua)
corDaAgua, posicaoDaAgua = estimarCorDaAgua(crua, canalEscuro)
normalizada = normalizarPelaCorDaAgua(crua, corDaAgua)
transmissao = calcularTransmissao(normalizada)
transmissaoRefinada = refinarTransmissao(transmissao, crua)
restaurada = recuperarCena(crua, corDaAgua, transmissaoRefinada)

imprimirCorDaAgua(corDaAgua, posicaoDaAgua)
imprimirResumoDaTransmissao("transmissão", transmissao)
imprimirResumoDaTransmissao("transmissão refinada", transmissaoRefinada)
imprimirResumoDaImagem("restaurada", restaurada)

salvarImagemNormalizada(canalEscuro, "1_canal_escuro", PASTA_SAIDA)
salvarImagemNormalizada(marcarPosicao(canalEscuro, posicaoDaAgua), "2_cor_da_agua", PASTA_SAIDA)
salvarImagemNormalizada(normalizada, "3a_normalizada", PASTA_SAIDA)
salvarImagemNormalizada(transmissao, "3b_transmissao", PASTA_SAIDA)
salvarImagemNormalizada(transmissaoRefinada, "3c_transmissao_refinada", PASTA_SAIDA)
salvarImagemNormalizada(restaurada, "3d_restaurada", PASTA_SAIDA)


# ================================ Parte 2 - Métodos Tradicionais ================================

tradicionalCor = aplicarMundoCinza(crua)
tradicionalContraste = aplicarClahe(tradicionalCor)
tradicional = aplicarFiltragemHomomorfica(tradicionalContraste)

mcglameryCor = aplicarMundoCinza(restaurada)
mcglameryContraste = aplicarClahe(mcglameryCor)
mcglamery = aplicarFiltragemHomomorfica(mcglameryContraste)

imprimirGanhos("Fluxo 1", ganhosDoMundoCinza(crua))
imprimirGanhos("McGlamery + métodos tradicionais", ganhosDoMundoCinza(restaurada))
imprimirResumoDaImagem("Fluxo 1, tradicional", tradicional)
imprimirResumoDaImagem("McGlamery + métodos tradicionais", mcglamery)

salvarImagemNormalizada(tradicionalCor, "trad_4_mundo_cinza", PASTA_SAIDA)
salvarImagemNormalizada(tradicionalContraste, "trad_5_clahe", PASTA_SAIDA)
salvarImagemNormalizada(tradicional, "trad_6_homomorfica", PASTA_SAIDA)
salvarImagemNormalizada(mcglameryCor, "mcg_4_mundo_cinza", PASTA_SAIDA)
salvarImagemNormalizada(mcglameryContraste, "mcg_5_clahe", PASTA_SAIDA)
salvarImagemNormalizada(mcglamery, "mcg_6_homomorfica", PASTA_SAIDA)


# ========================================== Avaliação ===========================================

imprimirNotas("original", crua, referencia)
imprimirNotas("Fluxo 1, tradicional", tradicional, referencia)
imprimirNotas("Fluxo 2, McGlamery", restaurada, referencia)
imprimirNotas("McGlamery + métodos tradicionais", mcglamery, referencia)
