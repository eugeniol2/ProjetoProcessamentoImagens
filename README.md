# Melhoria de imagens subaquáticas: métodos tradicionais × modelo de McGlamery-Jaffe

O projeto compara dois caminhos para melhorar uma foto subaquática da base UIEB:

- **Fluxo 1, tradicional:** corrige os sintomas na foto crua (cor, contraste e iluminação).
- **Fluxo 2, McGlamery:** desfaz a causa com o modelo físico de McGlamery-Jaffe. Estima a cor da
  água e a transmissão e recupera a cena sem o véu da água.

Cada saída é comparada, com PSNR e SSIM, com a imagem de referência da base, que foi escolhida
por voluntários como o melhor realce daquela foto. Também é avaliada a combinação
**McGlamery + métodos tradicionais**, que aplica o Fluxo 1 depois da restauração.

A comparação principal é feita na imagem piloto `13_img_.png`, na qual os parâmetros foram
escolhidos. Um segundo experimento repete a comparação nas 20 fotos da amostra que acompanha o
projeto, para ver como os mesmos parâmetros se comportam em outras fotos. Um terceiro
experimento testa combinações de parâmetros nas 890 fotos da base e fica com a que dá o maior
SSIM médio em cada fluxo.

Testado no Windows 11 com Python 3.11. Todos os comandos abaixo são para o **cmd**.

## 1. Preparar o ambiente

Abra o cmd na pasta do projeto e rode:

```bat
cd "C:\caminho\para\atividade_1"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

O projeto usa o `opencv-contrib-python-headless`: o módulo extra do OpenCV, onde fica o filtro
guiado, na versão sem interface gráfica, que também funciona em servidores como o Streamlit
Cloud. Esse pacote não pode ser instalado junto com outro pacote do OpenCV. Se o ambiente já
tiver o `opencv-python` ou o `opencv-contrib-python`, desinstale-os antes do `pip install`:

```bat
pip uninstall -y opencv-python opencv-contrib-python
```

Nas próximas vezes, basta ativar o ambiente com `.venv\Scripts\activate` antes de rodar os
scripts.

## 2. A base de dados

As fotos vêm da base UIEB (Li et al., 2019), que tem 890 fotos com referência.

A amostra fica na pasta `amostra/`, com as referências (33 MB): a imagem piloto
`13_img_.png` e as 19 fotos seguintes da base em ordem alfabética. A comparação principal é
feita na imagem piloto, e as outras 19 mostram como os mesmos parâmetros se comportam em fotos
diferentes. Os experimentos 1 e 2 e o app funcionam só com ela, logo depois de baixar o projeto.

O experimento 3 precisa da base completa. Baixe-a em
<https://li-chongyi.github.io/proj_benchmark.html> e coloque as pastas `raw-890` e
`reference-890` dentro de `dados/`. Com ela, o app também passa a mostrar as 890 fotos. O
resultado do experimento 1 não muda, porque a imagem piloto é a mesma. Os resultados do
experimento 3 já vêm no repositório, e o app os mostra mesmo sem a base completa.

## 3. Experimento 1: comparação na imagem piloto

```bat
python executar_exemplo.py
```

Leva alguns segundos. O script processa a `13_img_.png` pelos dois
fluxos e pela combinação, imprime o resumo de cada etapa e as notas, e grava uma imagem por
etapa em `resultados\exemplo\`. Essas imagens já vêm no repositório, geradas pelo autor, e
são regravadas a cada execução. As últimas linhas da saída são:

```
original: PSNR 16.94 dB | SSIM 0.733
Fluxo 1, tradicional: PSNR 20.32 dB | SSIM 0.890
Fluxo 2, McGlamery: PSNR 21.42 dB | SSIM 0.878
McGlamery + métodos tradicionais: PSNR 19.38 dB | SSIM 0.890
```

Imagens gravadas em `resultados\exemplo\`:

| Arquivo | Conteúdo |
|---|---|
| `0_original.png` e `referencia.png` | a foto crua e a referência da base |
| `1_canal_escuro.png` | etapa 1, o canal escuro |
| `2_cor_da_agua.png` | etapa 2, o pixel escolhido como cor da água (círculo vermelho) |
| `3a_normalizada.png` | etapa 3a, a foto dividida pela cor da água |
| `3b_transmissao.png` e `3c_transmissao_refinada.png` | etapas 3b e 3c, a transmissão antes e depois do filtro guiado |
| `3d_restaurada.png` | etapa 3d, a cena restaurada: a saída do Fluxo 2 |
| `trad_4_mundo_cinza.png`, `trad_5_clahe.png`, `trad_6_homomorfica.png` | etapas 4 a 6 do Fluxo 1. A última é a saída do Fluxo 1 |
| `mcg_4_mundo_cinza.png`, `mcg_5_clahe.png`, `mcg_6_homomorfica.png` | as mesmas etapas aplicadas à cena restaurada: a combinação McGlamery + métodos tradicionais |

## 4. Experimento 2: as 20 fotos da amostra

```bat
python executar_amostra.py
```

Leva cerca de 30 segundos. O script aplica os mesmos parâmetros às 20 fotos da `amostra/` e
grava as notas de cada foto em `resultados\amostra.csv`. Esse arquivo também já vem no
repositório. No fim, o script imprime as médias e em quantas fotos cada saída ficou à frente do
Fluxo 1:

```
foto crua: PSNR médio 15.78 dB | SSIM médio 0.728
Fluxo 1, tradicional: PSNR médio 16.96 dB | SSIM médio 0.817
Fluxo 2, McGlamery: PSNR médio 15.31 dB | SSIM médio 0.704
McGlamery + métodos tradicionais: PSNR médio 15.48 dB | SSIM médio 0.731
Fluxo 2, McGlamery melhor que o Fluxo 1: PSNR em 5 de 20 | SSIM em 4 de 20
McGlamery + métodos tradicionais melhor que o Fluxo 1: PSNR em 4 de 20 | SSIM em 2 de 20
```

Esse experimento sustenta a principal limitação do projeto. Os parâmetros escolhidos na imagem
piloto não funcionam bem em todas as fotos: a imagem piloto é um caso favorável à restauração.

## 5. Experimento 3: busca de parâmetros nas 890 fotos

```bat
python executar_busca_global.py
```

Precisa da base completa em `dados/`. O script testa combinações de valores para os parâmetros
de cada fluxo, aplica cada combinação a todas as fotos e fica com a de maior SSIM médio, a
mesma medida que o artigo do UIEB usa para comparar métodos. As fotos são divididas entre os
núcleos do processador. Num processador de 6 núcleos, a execução completa levou cerca de 3
horas: 20 minutos de exploração, 2h45 de busca principal e 10 minutos de combinação.

A busca tem três etapas:

1. **Exploração:** 729 combinações do Fluxo 1, com três valores para cada um dos seis
   parâmetros, em 20 fotos espalhadas pela base. Mostra quais parâmetros pesam e orienta os
   valores da etapa seguinte.
2. **Busca principal:** 144 combinações do Fluxo 1 e 45 do Fluxo 2, nas 890 fotos.
3. **Combinação:** McGlamery + métodos tradicionais, com os melhores valores dos dois fluxos,
   testando só o piso da transmissão.

Os valores testados ficam no topo do script (`VALORES_DA_EXPLORACAO`, `VALORES_DO_FLUXO_1`,
`VALORES_DO_FLUXO_2` e `VALORES_DA_COMBINACAO`). O resultado de cada foto é gravado assim que fica
pronto, em `resultados\busca_global\parciais\`. Se a execução for interrompida, rode o mesmo
comando de novo: o script pula as fotos já calculadas e continua de onde parou. Para um teste
rápido, passe a pasta da base e um número de fotos, por exemplo
`python executar_busca_global.py dados 10`.

No fim, o script imprime as melhores combinações e as médias nas 890 fotos. `projeto` são os
valores escolhidos na imagem piloto, `global` a melhor combinação para a base inteira e
`porFoto` a melhor combinação de cada foto, que serve de teto:

```
melhor do Fluxo 1: ganhoMaximo 1.3, limiteClahe 1.5, gradeClahe 1, ganhoBaixas 1, ganhoAltas 1.2, corte 30
melhor do Fluxo 2: janela 31, raio 15, piso 0.8
melhor da combinação: piso 0.9, com janela 31, raio 15, ganhoMaximo 1.3, limiteClahe 1.5, gradeClahe 1, ganhoBaixas 1, ganhoAltas 1.2, corte 30
crua: PSNR médio 17.36 dB | SSIM médio 0.768
fluxo1_projeto: PSNR médio 16.93 dB | SSIM médio 0.817
fluxo1_global: PSNR médio 19.67 dB | SSIM médio 0.861
fluxo1_porFoto: PSNR médio 20.82 dB | SSIM médio 0.898
fluxo2_projeto: PSNR médio 14.53 dB | SSIM médio 0.693
fluxo2_global: PSNR médio 17.30 dB | SSIM médio 0.777
fluxo2_porFoto: PSNR médio 18.15 dB | SSIM médio 0.813
combinacao_projeto: PSNR médio 15.00 dB | SSIM médio 0.728
combinacao_global: PSNR médio 19.55 dB | SSIM médio 0.858
```

Com os melhores valores, o Fluxo 1 continua à frente do Fluxo 2, e a restauração passa a
melhorar a foto crua na média. As últimas linhas comparam a escolha feita com todas as fotos com
a escolha feita só com as 200 primeiras, avaliadas nas outras 690. As duas dão notas parecidas,
o que indica que os valores escolhidos valem para fotos novas.

Tabelas gravadas em `resultados\busca_global\`:

| Arquivo | Conteúdo |
|---|---|
| `exploracao.csv` | PSNR e SSIM médios de cada combinação da exploração |
| `fluxo1.csv`, `fluxo2.csv`, `combinacao.csv` | PSNR e SSIM médios de cada combinação: em todas as fotos, nas 200 primeiras e nas outras 690 |
| `por_imagem.csv` | as notas de cada foto, com os valores do projeto, com os melhores valores e com a melhor combinação daquela foto |
| `melhores_por_foto.csv` | a melhor combinação de cada foto, em cada fluxo |

## 6. Opcional: o app interativo

```bat
python -m streamlit run app.py
```

Abre no navegador um passo a passo com a introdução, as etapas de cada fluxo, o resultado na
imagem piloto, os resultados globais do experimento 3 e os slides da apresentação. Em
cada etapa aparecem a imagem antes e depois, o problema que a etapa resolve e o motivo da escolha
do método. Os parâmetros podem ser mudados nos controles "Experimente", e qualquer foto da
amostra pode ser escolhida na barra lateral. Para encerrar, use `Ctrl+C` no cmd.

O botão "Abrir slides", no alto da barra lateral, mostra os slides da apresentação em duas abas,
e o botão "Voltar ao app" retorna à seção anterior. A aba Google Slides mostra a
[apresentação no Google Slides](https://docs.google.com/presentation/d/1rwOhLYTPyqACjoLne-jWar1hVlUJ9XGFF496TNmY9oc/edit?usp=sharing),
que também abre em modo apresentação pelo botão "Abrir no Google Slides". A aba PDF mostra o
arquivo `static\slides.pdf` um slide por vez, com botões para avançar e voltar, abre o PDF em
tela cheia no visualizador do navegador e permite baixá-lo. Para trocar o PDF, substitua esse
arquivo por outro com o mesmo nome.

## Organização do código

| Arquivo | O que faz |
|---|---|
| `utilitarios.py` | leitura e gravação de imagens (funciona com caminhos acentuados) |
| `base_de_dados.py` | lista as fotos e carrega cada foto crua com a sua referência |
| `parte1_restauracao.py` | Fluxo 2: as etapas 1 a 3d da restauração de McGlamery-Jaffe |
| `parte2_tradicional.py` | Fluxo 1: as etapas 4 a 6 dos métodos tradicionais |
| `avaliacao.py` | PSNR e SSIM contra a referência (scikit-image) |
| `executar_exemplo.py` | experimento 1, na imagem piloto |
| `executar_amostra.py` | experimento 2, nas 20 fotos da amostra |
| `executar_busca_global.py` | experimento 3, a busca de parâmetros nas 890 fotos |
| `app.py` | app interativo de estudo (Streamlit) |
| `requirements.txt` | versões dos pacotes |
| `amostra/` | 20 fotos da base UIEB com as referências |
| `resultados/` | as saídas dos três experimentos, geradas pelo autor |
| `static/slides.pdf` | os slides da apresentação, mostrados no app |
| `.streamlit/config.toml` | liga a entrega de arquivos da pasta `static/`, usada para abrir o PDF |

## As etapas e os parâmetros

O Fluxo 2 parte do modelo de McGlamery-Jaffe: cada pixel da foto I mistura a cena real J com a
cor da água B, na proporção dada pela transmissão t.

```
I = J · t + B · (1 − t)
```

As etapas 1 a 3c estimam B e t a partir da própria foto, e a etapa 3d faz a conta ao contrário
para recuperar J.

| Etapa | Função | O que faz | Parâmetro |
|---|---|---|---|
| 1. Canal escuro | `calcularCanalEscuro` | menor valor entre verde e azul, seguido de uma erosão (`cv2.erode`) | janela 15 × 15 |
| 2. Cor da água B | `estimarCorDaAgua` | entre os 0,1% pixels mais claros do canal escuro, no terço de cima e sem pixels quase brancos, o mais claro na foto | — |
| 3a. Normalização | `normalizarPelaCorDaAgua` | divide cada canal pela cor da água | — |
| 3b. Transmissão | `calcularTransmissao` | 1 menos o canal escuro da foto normalizada | janela 15 × 15 |
| 3c. Refinamento | `refinarTransmissao` | filtro guiado (`cv2.ximgproc.guidedFilter`), com a foto como guia | raio 30, ε 0,001 |
| 3d. Recuperação de J | `recuperarCena` | afasta cada pixel da cor da água, dividindo pela transmissão | piso t0 = 0,5 |
| 4. Cor | `aplicarMundoCinza` | mundo cinza (gray world) | ganho máximo 1,3 |
| 5. Contraste | `aplicarClahe` | CLAHE no canal L do LAB (`cv2.createCLAHE`) | limite 2, grade 8 × 8 |
| 6. Iluminação | `aplicarFiltragemHomomorfica` | filtragem homomórfica no canal L, com a FFT do NumPy | ganhos 0,8 e 1,2, corte 30 |

Os parâmetros são as constantes no topo de `parte1_restauracao.py` e `parte2_tradicional.py`.
Para testar outros valores, altere essas constantes e rode os scripts de novo, ou use os
controles do app. As constantes guardam os valores escolhidos na imagem piloto. Os melhores
valores para a base inteira, encontrados no experimento 3, estão nas tabelas de
`resultados\busca_global\` e na seção Resultados globais do app.
