from skimage.metrics import peak_signal_noise_ratio, structural_similarity

FAIXA_DOS_DADOS = 1.0
JANELA_SSIM = 7
EIXO_DOS_CANAIS = 2


# ============================================= PSNR =============================================


def calcularPsnr(imagem, referencia):
    return peak_signal_noise_ratio(referencia, imagem, data_range=FAIXA_DOS_DADOS)


# ============================================= SSIM =============================================


def calcularSsim(imagem, referencia):
    return structural_similarity(
        referencia,
        imagem,
        win_size=JANELA_SSIM,
        data_range=FAIXA_DOS_DADOS,
        channel_axis=EIXO_DOS_CANAIS,
    )
