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
projeto, para ver como os mesmos parâmetros se comportam em outras fotos.

Testado no Windows 11 com Python 3.11. Todos os comandos abaixo são para o **cmd**.

## 1. Preparar o ambiente

Abra o cmd na pasta do projeto e rode:

```bat
cd "C:\caminho\para\atividade_1"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

O projeto usa o `opencv-contrib-python`, porque o filtro guiado fica no módulo extra do OpenCV.
Esse pacote não pode ser instalado junto com o `opencv-python`. Se o ambiente já tiver o
`opencv-python`, desinstale-o antes do `pip install`:

```bat
pip uninstall -y opencv-python
```

Nas próximas vezes, basta ativar o ambiente com `.venv\Scripts\activate` antes de rodar os
scripts.

## 2. A base de dados

As fotos vêm da base UIEB (Li et al., 2019), que tem 890 fotos com referência. Para este
projeto, uma amostra de 20 fotos é suficiente, e optamos por não usar a base completa.

A amostra fica na pasta `amostra/`, com as referências (33 MB): a imagem piloto
`13_img_.png` e as 19 fotos seguintes da base em ordem alfabética. A comparação principal é
feita na imagem piloto, e as outras 19 mostram como os mesmos parâmetros se comportam em fotos
diferentes. Os dois experimentos e o app funcionam só com ela, logo depois de baixar o projeto.

Quem quiser explorar outras fotos no app pode baixar a base completa em
<https://li-chongyi.github.io/proj_benchmark.html> e colocar as pastas `raw-890` e
`reference-890` dentro de `dados/`. O app e o experimento 1 passam a usá-la. O resultado do
experimento 1 não muda, porque a imagem piloto é a mesma.

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

## 5. Opcional: o app interativo

```bat
python -m streamlit run app.py
```

Abre no navegador um passo a passo com a introdução, as etapas de cada fluxo e o resultado. Em
cada etapa aparecem a imagem antes e depois, o problema que a etapa resolve e o motivo da escolha
do método. Os parâmetros podem ser mudados nos controles "Experimente", e qualquer foto da
amostra pode ser escolhida na barra lateral. Para encerrar, use `Ctrl+C` no cmd.

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
| `app.py` | app interativo de estudo (Streamlit) |
| `requirements.txt` | versões dos pacotes |
| `amostra/` | 20 fotos da base UIEB com as referências |
| `resultados/` | as saídas dos dois experimentos, geradas pelo autor |

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
controles do app.
