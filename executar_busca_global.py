import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import parte1_restauracao as parte1
import parte2_tradicional as parte2
from avaliacao import calcularPsnr, calcularSsim
from base_de_dados import PASTA_CRUAS, carregarPar, listarNomesDasImagens, pastaDaBase

PASTA_PROJETO = Path(__file__).parent
PASTA_SAIDA = PASTA_PROJETO / "resultados" / "busca_global"
PASTA_PARCIAIS = PASTA_SAIDA / "parciais"
PROCESSOS = max(1, (os.cpu_count() or 1) - 2)
INTERVALO_DO_PROGRESSO = 10
TAMANHO_AJUSTE = 200
PSNR = 0
SSIM = 1
FASE_EXPLORACAO = "exploracao"
FASE_1 = "fase1"
PASSO_DA_EXPLORACAO = 44
FOTOS_DA_EXPLORACAO = 20
VALORES_DA_EXPLORACAO = {
    "ganhoMaximo": [1.0, 1.3, 1.6],
    "limiteClahe": [1.0, 2.0, 3.0],
    "gradeClahe": [4, 8, 16],
    "ganhoBaixas": [0.6, 0.8, 1.0],
    "ganhoAltas": [1.0, 1.2, 1.4],
    "corte": [15, 30, 60],
}
VALORES_DO_FLUXO_1 = {
    "ganhoMaximo": [1.3, 1.6, 2.0],
    "limiteClahe": [1.5, 2.0, 2.5],
    "gradeClahe": [1, 2, 4, 8],
    "ganhoBaixas": [0.8, 1.0],
    "ganhoAltas": [1.0, 1.2],
    "corte": [30],
}
VALORES_DO_FLUXO_2 = {
    "janela": [7, 15, 31],
    "raio": [15, 30, 60],
    "piso": [0.5, 0.6, 0.7, 0.8, 0.9],
}
VALORES_DA_COMBINACAO = {"piso": VALORES_DO_FLUXO_2["piso"]}
PROJETO_FLUXO_1 = {
    "ganhoMaximo": parte2.GANHO_MAXIMO,
    "limiteClahe": parte2.LIMITE_CONTRASTE_CLAHE,
    "gradeClahe": parte2.BLOCOS_CLAHE,
    "ganhoBaixas": parte2.GANHO_BAIXAS_FREQUENCIAS,
    "ganhoAltas": parte2.GANHO_ALTAS_FREQUENCIAS,
    "corte": parte2.FREQUENCIA_DE_CORTE,
}
PROJETO_FLUXO_2 = {
    "janela": parte1.JANELA_CANAL_ESCURO,
    "raio": parte1.RAIO_FILTRO_GUIADO,
    "piso": parte1.TRANSMISSAO_MINIMA,
}
SAIDAS = {
    "crua": "foto crua",
    "fluxo1": "Fluxo 1, tradicional",
    "fluxo2": "Fluxo 2, McGlamery",
    "combinacao": "McGlamery + métodos tradicionais",
}


def combinacoes(valores):
    return [dict(zip(valores, escolha)) for escolha in itertools.product(*valores.values())]


def prepararReferencias(referencia):
    return referencia, referencia.astype(np.float32)


def notasContraReferencia(imagem, referencia, referencia32):
    psnr = calcularPsnr(imagem, referencia)
    ssim = calcularSsim(imagem.astype(np.float32), referencia32)

    return psnr, ssim


# ================================ Fluxo 1 - Métodos Tradicionais ================================


def espectroDaLuminosidade(lab):
    luminosidade = lab[:, :, parte2.LUMINOSIDADE]

    return parte2.espectroCentralizado(parte2.paraDominioLogaritmico(luminosidade))


def filtrarEspectro(lab, espectro, filtro):
    filtrada = parte2.voltarAoDominioEspacial(espectro * filtro)
    luminosidade = parte2.paraNiveisDeLuminosidade(parte2.sairDoDominioLogaritmico(filtrada))

    return parte2.labParaRgb(parte2.trocarLuminosidade(lab, luminosidade))


def filtrosHomomorficos(altura, largura, valores):
    ganhosECortes = itertools.product(
        valores["ganhoBaixas"], valores["ganhoAltas"], valores["corte"]
    )

    return [
        parte2.filtroEnfaseAltasFrequencias(altura, largura, ganhoBaixas, ganhoAltas, corte)
        for ganhoBaixas, ganhoAltas, corte in ganhosECortes
    ]


def notasDoContraste(contraste, filtros, referencias):
    lab = parte2.rgbParaLab(contraste)
    espectro = espectroDaLuminosidade(lab)

    return [
        notasContraReferencia(filtrarEspectro(lab, espectro, filtro), *referencias)
        for filtro in filtros
    ]


def notasDaCor(cor, filtros, referencias, valores):
    notas = []
    for limite, blocos in itertools.product(valores["limiteClahe"], valores["gradeClahe"]):
        notas += notasDoContraste(parte2.aplicarClahe(cor, limite, blocos), filtros, referencias)

    return notas


def notasDoFluxo1(crua, referencias, valores):
    filtros = filtrosHomomorficos(*crua.shape[:2], valores)
    notas = []
    for ganhoMaximo in valores["ganhoMaximo"]:
        cor = parte2.aplicarMundoCinza(crua, ganhoMaximo)
        notas += notasDaCor(cor, filtros, referencias, valores)

    return np.array(notas)


def aplicarTradicionais(imagem, parametros):
    cor = parte2.aplicarMundoCinza(imagem, parametros["ganhoMaximo"])
    contraste = parte2.aplicarClahe(cor, parametros["limiteClahe"], parametros["gradeClahe"])

    return parte2.aplicarFiltragemHomomorfica(
        contraste, parametros["ganhoBaixas"], parametros["ganhoAltas"], parametros["corte"]
    )


# ================================== Fluxo 2 - Restauração McGlamery =============================


def variaveisDoModelo(crua, janela):
    canalEscuro = parte1.calcularCanalEscuro(crua, janela)
    corDaAgua, _ = parte1.estimarCorDaAgua(crua, canalEscuro)
    normalizada = parte1.normalizarPelaCorDaAgua(crua, corDaAgua)

    return corDaAgua, parte1.calcularTransmissao(normalizada, janela)


def cenasRestauradas(crua):
    for janela in VALORES_DO_FLUXO_2["janela"]:
        corDaAgua, transmissao = variaveisDoModelo(crua, janela)
        for raio in VALORES_DO_FLUXO_2["raio"]:
            refinada = parte1.refinarTransmissao(transmissao, crua, raio)
            for piso in VALORES_DO_FLUXO_2["piso"]:
                yield parte1.recuperarCena(crua, corDaAgua, refinada, piso)


def restaurar(crua, parametros):
    corDaAgua, transmissao = variaveisDoModelo(crua, parametros["janela"])
    refinada = parte1.refinarTransmissao(transmissao, crua, parametros["raio"])

    return parte1.recuperarCena(crua, corDaAgua, refinada, parametros["piso"])


def notasDoFluxo2(crua, referencias):
    return np.array([notasContraReferencia(cena, *referencias) for cena in cenasRestauradas(crua)])


def notasDaCombinacao(crua, referencias, tradicional, restauracao):
    corDaAgua, transmissao = variaveisDoModelo(crua, restauracao["janela"])
    refinada = parte1.refinarTransmissao(transmissao, crua, restauracao["raio"])
    cenas = (
        parte1.recuperarCena(crua, corDaAgua, refinada, piso)
        for piso in VALORES_DA_COMBINACAO["piso"]
    )

    return np.array(
        [
            notasContraReferencia(aplicarTradicionais(cena, tradicional), *referencias)
            for cena in cenas
        ]
    )


def notasDaCombinacaoDoProjeto(crua, referencias):
    cena = restaurar(crua, PROJETO_FLUXO_2)

    return np.array(notasContraReferencia(aplicarTradicionais(cena, PROJETO_FLUXO_1), *referencias))


# ===================================== Execução em Paralelo =====================================


def iniciarProcesso():
    cv2.setNumThreads(1)


def nomeDaFase2(tradicional, restauracao):
    valores = {**tradicional, "janela": restauracao["janela"], "raio": restauracao["raio"]}

    return "fase2_" + "_".join(f"{nome}{valor:g}" for nome, valor in valores.items())


def arquivoParcial(fase, nomeImagem):
    return PASTA_PARCIAIS / fase / f"{Path(nomeImagem).stem}.npz"


def salvarParcial(arquivo, **notas):
    temporario = arquivo.with_name(arquivo.stem + "_temporario.npz")
    np.savez(temporario, **notas)
    os.replace(temporario, arquivo)


def avaliarExploracao(tarefa):
    nomeImagem, pastaBase, fase = tarefa
    crua, referencia = carregarPar(nomeImagem, pastaBase)
    referencias = prepararReferencias(referencia)
    salvarParcial(
        arquivoParcial(fase, nomeImagem),
        fluxo1=notasDoFluxo1(crua, referencias, VALORES_DA_EXPLORACAO),
    )

    return nomeImagem


def avaliarFase1(tarefa):
    nomeImagem, pastaBase, fase = tarefa
    crua, referencia = carregarPar(nomeImagem, pastaBase)
    referencias = prepararReferencias(referencia)
    salvarParcial(
        arquivoParcial(fase, nomeImagem),
        crua=np.array(notasContraReferencia(crua, *referencias)),
        fluxo1=notasDoFluxo1(crua, referencias, VALORES_DO_FLUXO_1),
        fluxo2=notasDoFluxo2(crua, referencias),
    )

    return nomeImagem


def avaliarFase2(tarefa):
    nomeImagem, pastaBase, fase, tradicional, restauracao = tarefa
    crua, referencia = carregarPar(nomeImagem, pastaBase)
    referencias = prepararReferencias(referencia)
    salvarParcial(
        arquivoParcial(fase, nomeImagem),
        combinacao=notasDaCombinacao(crua, referencias, tradicional, restauracao),
        combinacaoDoProjeto=notasDaCombinacaoDoProjeto(crua, referencias),
    )

    return nomeImagem


def imprimirProgresso(fase, feitas, total, inicio):
    if feitas % INTERVALO_DO_PROGRESSO and feitas != total:
        return
    minutos = (time.perf_counter() - inicio) / 60
    restantes = minutos / feitas * (total - feitas)
    print(
        f"{fase}: {feitas} de {total} fotos | {minutos:.0f} min | faltam cerca de "
        f"{restantes:.0f} min",
        flush=True,
    )


def executarFase(funcao, tarefas):
    fase = tarefas[0][2]
    (PASTA_PARCIAIS / fase).mkdir(parents=True, exist_ok=True)
    pendentes = [tarefa for tarefa in tarefas if not arquivoParcial(fase, tarefa[0]).exists()]
    print(
        f"{fase}: {len(tarefas) - len(pendentes)} fotos já calculadas, {len(pendentes)} a calcular"
    )
    if not pendentes:
        return
    inicio = time.perf_counter()
    with Pool(PROCESSOS, initializer=iniciarProcesso) as grupo:
        for feitas, _ in enumerate(grupo.imap_unordered(funcao, pendentes), start=1):
            imprimirProgresso(fase, feitas, len(pendentes), inicio)


# ======================================= Tabelas e Resumo =======================================


def lerParcial(fase, nomeImagem, chave):
    with np.load(arquivoParcial(fase, nomeImagem)) as dados:
        return dados[chave]


def lerNotas(fase, nomes, chave):
    return np.stack([lerParcial(fase, nome, chave) for nome in nomes])


def media(valores):
    return round(float(valores.mean()), 6) if len(valores) else float("nan")


def mediasDasNotas(notas):
    ajuste, avaliacao = notas[:TAMANHO_AJUSTE], notas[TAMANHO_AJUSTE:]

    return {
        "psnr_media": media(notas[:, PSNR]),
        "ssim_media": media(notas[:, SSIM]),
        "psnr_ajuste": media(ajuste[:, PSNR]),
        "ssim_ajuste": media(ajuste[:, SSIM]),
        "psnr_avaliacao": media(avaliacao[:, PSNR]),
        "ssim_avaliacao": media(avaliacao[:, SSIM]),
    }


def mediasDaExploracao(notas):
    return {"psnr_media": media(notas[:, PSNR]), "ssim_media": media(notas[:, SSIM])}


def tabelaDeCombinacoes(valores, notas, fixos, calcularMedias=mediasDasNotas):
    return [
        {**combinacao, **fixos, **calcularMedias(notas[:, indice])}
        for indice, combinacao in enumerate(combinacoes(valores))
    ]


def fotosDaExploracao(nomes):
    return nomes[::PASSO_DA_EXPLORACAO][:FOTOS_DA_EXPLORACAO]


def indiceDaMelhor(notas):
    return int(np.argmax(notas[:, :, SSIM].mean(axis=0)))


def melhoresDaFoto(notas):
    return notas[np.arange(len(notas)), np.argmax(notas[:, :, SSIM], axis=1)]


def tabelaPorImagem(nomes, saidas):
    linhas = []
    for posicao, nomeImagem in enumerate(nomes):
        linha = {"imagem": nomeImagem}
        for rotulo, notas in saidas.items():
            linha[f"psnr_{rotulo}"] = round(float(notas[posicao, PSNR]), 6)
            linha[f"ssim_{rotulo}"] = round(float(notas[posicao, SSIM]), 6)
        linhas.append(linha)

    return linhas


def valoresDaMelhorDaFoto(valores, notasDaFoto):
    return combinacoes(valores)[int(np.argmax(notasDaFoto[:, SSIM]))]


def tabelaDosMelhoresPorFoto(nomes, notasFluxo1, notasFluxo2):
    return [
        {
            "imagem": nomeImagem,
            **valoresDaMelhorDaFoto(VALORES_DO_FLUXO_1, notasFluxo1[posicao]),
            **valoresDaMelhorDaFoto(VALORES_DO_FLUXO_2, notasFluxo2[posicao]),
        }
        for posicao, nomeImagem in enumerate(nomes)
    ]


def salvarTabela(linhas, nomeArquivo):
    caminho = PASTA_SAIDA / nomeArquivo
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"tabela salva em {caminho}")


def textoDosValores(valores):
    return ", ".join(f"{nome} {valor:g}" for nome, valor in valores.items())


def imprimirMedias(rotulo, notas):
    print(
        f"{rotulo}: PSNR médio {notas[:, PSNR].mean():.2f} dB | "
        f"SSIM médio {notas[:, SSIM].mean():.3f}"
    )


def imprimirVerificacao(rotulo, notas):
    if len(notas) <= TAMANHO_AJUSTE:
        return
    escolhidaNoAjuste = int(np.argmax(notas[:TAMANHO_AJUSTE, :, SSIM].mean(axis=0)))
    avaliacao = notas[TAMANHO_AJUSTE:, :, SSIM].mean(axis=0)
    print(
        f"{rotulo}: escolhida só com as {TAMANHO_AJUSTE} primeiras fotos, a melhor combinação dá "
        f"SSIM {avaliacao[escolhidaNoAjuste]:.3f} nas outras; a escolhida com todas dá "
        f"{avaliacao[indiceDaMelhor(notas)]:.3f}"
    )


# ================================================================================================

if __name__ == "__main__":
    pastaBase = Path(sys.argv[1]) if len(sys.argv) > 1 else pastaDaBase(PASTA_PROJETO)
    quantidadeDeFotos = int(sys.argv[2]) if len(sys.argv) > 2 else None
    nomes = listarNomesDasImagens(pastaBase / PASTA_CRUAS)[:quantidadeDeFotos]
    print(f"base usada: {pastaBase} ({len(nomes)} fotos) | {PROCESSOS} processos")

    nomesDaExploracao = fotosDaExploracao(nomes)
    executarFase(
        avaliarExploracao, [(nome, pastaBase, FASE_EXPLORACAO) for nome in nomesDaExploracao]
    )
    notasDaExploracao = lerNotas(FASE_EXPLORACAO, nomesDaExploracao, "fluxo1")
    salvarTabela(
        tabelaDeCombinacoes(VALORES_DA_EXPLORACAO, notasDaExploracao, {}, mediasDaExploracao),
        "exploracao.csv",
    )

    executarFase(avaliarFase1, [(nome, pastaBase, FASE_1) for nome in nomes])

    notasCrua = lerNotas(FASE_1, nomes, "crua")
    notasFluxo1 = lerNotas(FASE_1, nomes, "fluxo1")
    notasFluxo2 = lerNotas(FASE_1, nomes, "fluxo2")
    melhorFluxo1 = indiceDaMelhor(notasFluxo1)
    melhorFluxo2 = indiceDaMelhor(notasFluxo2)
    tradicional = combinacoes(VALORES_DO_FLUXO_1)[melhorFluxo1]
    restauracao = combinacoes(VALORES_DO_FLUXO_2)[melhorFluxo2]
    print(f"melhor do Fluxo 1: {textoDosValores(tradicional)}")
    print(f"melhor do Fluxo 2: {textoDosValores(restauracao)}")

    faseDaCombinacao = nomeDaFase2(tradicional, restauracao)
    executarFase(
        avaliarFase2,
        [(nome, pastaBase, faseDaCombinacao, tradicional, restauracao) for nome in nomes],
    )

    notasCombinacao = lerNotas(faseDaCombinacao, nomes, "combinacao")
    notasCombinacaoDoProjeto = lerNotas(faseDaCombinacao, nomes, "combinacaoDoProjeto")
    melhorCombinacao = indiceDaMelhor(notasCombinacao)
    fixosDaCombinacao = {
        "janela": restauracao["janela"],
        "raio": restauracao["raio"],
        **tradicional,
    }
    print(
        "melhor da combinação: "
        f"{textoDosValores(combinacoes(VALORES_DA_COMBINACAO)[melhorCombinacao])}, com "
        f"{textoDosValores(fixosDaCombinacao)}"
    )

    saidas = {
        "crua": notasCrua,
        "fluxo1_projeto": notasFluxo1[:, combinacoes(VALORES_DO_FLUXO_1).index(PROJETO_FLUXO_1)],
        "fluxo1_global": notasFluxo1[:, melhorFluxo1],
        "fluxo1_porFoto": melhoresDaFoto(notasFluxo1),
        "fluxo2_projeto": notasFluxo2[:, combinacoes(VALORES_DO_FLUXO_2).index(PROJETO_FLUXO_2)],
        "fluxo2_global": notasFluxo2[:, melhorFluxo2],
        "fluxo2_porFoto": melhoresDaFoto(notasFluxo2),
        "combinacao_projeto": notasCombinacaoDoProjeto,
        "combinacao_global": notasCombinacao[:, melhorCombinacao],
    }

    salvarTabela(tabelaDeCombinacoes(VALORES_DO_FLUXO_1, notasFluxo1, {}), "fluxo1.csv")
    salvarTabela(tabelaDeCombinacoes(VALORES_DO_FLUXO_2, notasFluxo2, {}), "fluxo2.csv")
    salvarTabela(
        tabelaDeCombinacoes(VALORES_DA_COMBINACAO, notasCombinacao, fixosDaCombinacao),
        "combinacao.csv",
    )
    salvarTabela(tabelaPorImagem(nomes, saidas), "por_imagem.csv")
    salvarTabela(tabelaDosMelhoresPorFoto(nomes, notasFluxo1, notasFluxo2), "melhores_por_foto.csv")

    for rotulo, notas in saidas.items():
        imprimirMedias(rotulo, notas)
    imprimirVerificacao(SAIDAS["fluxo1"], notasFluxo1)
    imprimirVerificacao(SAIDAS["fluxo2"], notasFluxo2)
    imprimirVerificacao(SAIDAS["combinacao"], notasCombinacao)
