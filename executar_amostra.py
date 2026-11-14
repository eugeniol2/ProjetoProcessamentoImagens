import csv
import sys
from pathlib import Path

import numpy as np

from avaliacao import calcularPsnr, calcularSsim
from base_de_dados import PASTA_AMOSTRA, PASTA_CRUAS, carregarPar, listarNomesDasImagens
from parte1_restauracao import (
    calcularCanalEscuro,
    calcularTransmissao,
    estimarCorDaAgua,
    normalizarPelaCorDaAgua,
    recuperarCena,
    refinarTransmissao,
)
from parte2_tradicional import aplicarClahe, aplicarFiltragemHomomorfica, aplicarMundoCinza

PASTA_PROJETO = Path(__file__).parent
PASTA_BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_PROJETO / PASTA_AMOSTRA
ARQUIVO_SAIDA = PASTA_PROJETO / "resultados" / "amostra.csv"
SAIDAS = {
    "crua": "foto crua",
    "fluxo1": "Fluxo 1, tradicional",
    "fluxo2": "Fluxo 2, McGlamery",
    "combinacao": "McGlamery + métodos tradicionais",
}


def restaurar(crua):
    canalEscuro = calcularCanalEscuro(crua)
    corDaAgua, _ = estimarCorDaAgua(crua, canalEscuro)
    normalizada = normalizarPelaCorDaAgua(crua, corDaAgua)
    transmissao = refinarTransmissao(calcularTransmissao(normalizada), crua)

    return recuperarCena(crua, corDaAgua, transmissao)


def aplicarTradicionais(imagem):
    return aplicarFiltragemHomomorfica(aplicarClahe(aplicarMundoCinza(imagem)))


def calcularSaidas(crua):
    restaurada = restaurar(crua)

    return {
        "crua": crua,
        "fluxo1": aplicarTradicionais(crua),
        "fluxo2": restaurada,
        "combinacao": aplicarTradicionais(restaurada),
    }


def avaliarImagem(nomeImagem, pastaBase):
    crua, referencia = carregarPar(nomeImagem, pastaBase)
    linha = {"imagem": nomeImagem}
    for chave, imagem in calcularSaidas(crua).items():
        linha[f"psnr_{chave}"] = calcularPsnr(imagem, referencia)
        linha[f"ssim_{chave}"] = calcularSsim(imagem, referencia)

    return linha


def salvarTabela(linhas, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def coluna(linhas, nome):
    return np.array([linha[nome] for linha in linhas])


def imprimirMedias(linhas):
    for chave, rotulo in SAIDAS.items():
        psnr = coluna(linhas, f"psnr_{chave}").mean()
        ssim = coluna(linhas, f"ssim_{chave}").mean()
        print(f"{rotulo}: PSNR médio {psnr:.2f} dB | SSIM médio {ssim:.3f}")


def contarVitorias(linhas, chave, metrica):
    return int(np.sum(coluna(linhas, f"{metrica}_{chave}") > coluna(linhas, f"{metrica}_fluxo1")))


def imprimirVitorias(linhas):
    total = len(linhas)
    for chave in ["fluxo2", "combinacao"]:
        vitoriasPsnr = contarVitorias(linhas, chave, "psnr")
        vitoriasSsim = contarVitorias(linhas, chave, "ssim")
        print(
            f"{SAIDAS[chave]} melhor que o Fluxo 1: PSNR em {vitoriasPsnr} de {total} | "
            f"SSIM em {vitoriasSsim} de {total}"
        )


# ================================================================================================

nomesDaAmostra = listarNomesDasImagens(PASTA_BASE / PASTA_CRUAS)
print(f"base usada: {PASTA_BASE} ({len(nomesDaAmostra)} fotos)")

linhas = [avaliarImagem(nomeImagem, PASTA_BASE) for nomeImagem in nomesDaAmostra]

salvarTabela(linhas, ARQUIVO_SAIDA)
print(f"tabela salva em {ARQUIVO_SAIDA}")
imprimirMedias(linhas)
imprimirVitorias(linhas)
