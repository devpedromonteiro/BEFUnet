# BEFUnet: A Hybrid CNN-Transformer Architecture for Precise Medical Image Segmentation

:closed_book: [[arxiv]](https://arxiv.org/abs/2402.08793)

# :tada: :tada: :tada: News

- **`2024/03/20` First release.**


## Train & Test --- Synapse Dataset
Please go to ["Colab_BEFUnet.ipynb"](https://github.com/devpedromonteiro/BEFUnet/blob/combination/Colab_BEFUnet.ipynb) for complete detail on dataset preparation and Train/Test procedure or follow the instructions below. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devpedromonteiro/BEFUnet/blob/combination/Colab_BEFUnet.ipynb)

## 

![images](Figures/models.png)

# Usage

This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code: Python 3, Pytorch

## Installation
1) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

2) Run the below code to train BEFUnet on the synapse dataset.

    ```bash
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5  --model_name BEFUnet --batch_size 10 --eval_interval 20 --max_epochs 500 
   ```

3) Run the below code to test BEFUnet on the synapse dataset.
    ```bash
    python test.py --test_path ./data/Synapse/test_vol_h5 --model_name BEFUnet --is_savenii --model_weight 
    ```


## Linformer Attention Implementation

### O que foi implementado?

Foi adicionada uma implementação completa do **Linformer** como alternativa eficiente aos mecanismos de atenção padrão do BEFUnet. O Linformer é uma variante de atenção linear que reduz significativamente a complexidade computacional de O(N²) para O(N×k), onde k << N é a dimensão de projeção.

#### Componentes Implementados:

1. **`LinformerWindowAttention`** (`utils.py`):
   - Versão Linformer do `WindowAttention` usado nos blocos Swin Transformer
   - Projeta keys e values para uma dimensão menor (k) antes do cálculo de atenção
   - Mantém compatibilidade com relative position bias
   - Reduz uso de memória e acelera o processamento em janelas de atenção

2. **`LinformerCrossAttention`** (`utils.py`):
   - Versão Linformer do `CrossAttention` usado no módulo DLF (Dual-Level Fusion)
   - Suporta sequências de comprimento variável
   - Usa matrizes de projeção parametrizadas para diferentes comprimentos de sequência

3. **Integração nos Blocos**:
   - `SwinTransformerBlock`: Suporta escolha entre atenção padrão e Linformer
   - `CrossAttentionBlock`: Suporta escolha entre atenção padrão e Linformer
   - `BasicLayer` e `MultiScaleBlock`: Propagam parâmetros do Linformer através da arquitetura

4. **Configuração** (`configs/BEFUnet_configs.py`):
   - `use_linformer`: Flag para habilitar/desabilitar Linformer (padrão: False)
   - `linformer_k`: Dimensão de projeção k (padrão: 64)
   - `linformer_max_seq_len`: Comprimento máximo de sequência para CrossAttention (padrão: 512)

### Por que foi implementado?

A implementação do Linformer foi realizada com base nas seguintes motivações:

1. **Eficiência Computacional**: 
   - O mecanismo de atenção padrão tem complexidade quadrática O(N²) em relação ao comprimento da sequência
   - Para imagens médicas de alta resolução, isso pode resultar em uso excessivo de memória e tempo de treinamento
   - O Linformer reduz isso para O(N×k), onde k é tipicamente muito menor que N (ex: k=64 vs N=3136)

2. **Estudos de Ablação**:
   - Conforme sugerido em revisões, é importante avaliar variantes leves de atenção
   - Permite comparar o desempenho entre atenção padrão e atenção linear
   - Facilita experimentos para encontrar o melhor trade-off entre precisão e eficiência

3. **Escalabilidade**:
   - Permite treinar modelos com imagens maiores ou batches maiores
   - Reduz o uso de GPU/TPU, tornando o modelo mais acessível
   - Facilita a aplicação em ambientes com recursos limitados

4. **Alternativa ao Flash Attention**:
   - Flash Attention-2 foi tentado mas não funcionou neste contexto
   - Linformer oferece uma alternativa eficiente e funcional
   - Implementação mais simples e compatível com a arquitetura existente

### Como usar?

Para habilitar o Linformer, configure no arquivo de configuração:

```python
from configs.BEFUnet_configs import get_BEFUnet_configs

config = get_BEFUnet_configs()
config.use_linformer = True      # Habilita Linformer
config.linformer_k = 64          # Dimensão de projeção (ajuste conforme necessário)
config.linformer_max_seq_len = 512  # Para CrossAttention
```

O modelo automaticamente usará atenção linear (Linformer) em vez da atenção padrão quando `use_linformer=True`.

**Nota importante sobre pesos pré-treinados:**
- Quando `use_linformer=True`, os pesos de atenção (qkv, proj) não podem ser carregados do checkpoint pré-treinado do Swin Transformer, pois a arquitetura é diferente
- O modelo carrega automaticamente com `strict=False`, mantendo os pesos compatíveis (normalização, MLP, etc.) e inicializando aleatoriamente os pesos de atenção e as matrizes de projeção do Linformer (E_k, E_v)
- Isso significa que quando usar Linformer pela primeira vez, o modelo começará do zero para os parâmetros de atenção, mas manterá os pesos pré-treinados das outras partes da arquitetura

### Troubleshooting

**Problema**: Erro ao carregar pesos pré-treinados quando `use_linformer=True`
- **Solução**: Isso é esperado e já está tratado automaticamente. O modelo usa `strict=False` para carregar apenas pesos compatíveis. Os pesos de atenção serão inicializados aleatoriamente.

**Problema**: Erro de dimensões ao aplicar máscaras com Linformer
- **Solução**: Já corrigido na implementação. A máscara é adaptada automaticamente para a estrutura linear do Linformer.

**Problema**: Performance inicial pior com Linformer habilitado
- **Explicação**: Isso é esperado, pois os pesos de atenção começam do zero. O modelo precisa ser treinado do início ou fazer fine-tuning. Os pesos pré-treinados das outras partes (normalização, MLP) ainda são carregados.

### Referências

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
- [BEFUnet Paper](https://arxiv.org/abs/2402.08793)

## Acknowledgement
We borrowed the code from [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [PiDinet](https://github.com/hellozhuo/pidinet). Thanks for their wonderful works.


## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```
@article{manzari2024befunet,
  title={BEFUnet: A Hybrid CNN-Transformer Architecture for Precise Medical Image Segmentation},
  author={Manzari, Omid Nejati and Kaleybar, Javad Mirzapour and Saadat, Hooman and Maleki, Shahin},
  journal={arXiv preprint arXiv:2402.08793},
  year={2024}
}
```

