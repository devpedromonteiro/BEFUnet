# Implementação do Linformer no BEFUnet

## Visão Geral

Este documento descreve em detalhes a implementação do mecanismo de atenção Linformer no BEFUnet, uma arquitetura híbrida CNN-Transformer para segmentação de imagens médicas.

## Contexto e Motivação

### Problema Original

O mecanismo de atenção padrão em Transformers tem complexidade quadrática O(N²) em relação ao comprimento da sequência N. No contexto do BEFUnet:

- **WindowAttention**: Para uma janela de 7×7, temos N=49 tokens, resultando em uma matriz de atenção de 49×49
- **CrossAttention**: Para patches de 56×56, temos N=3136 tokens, resultando em uma matriz de 3136×3136

Isso causa:
- Alto consumo de memória GPU
- Tempo de treinamento prolongado
- Limitações no tamanho de batch ou resolução de imagem

### Solução: Linformer

O Linformer propõe projetar as matrizes de keys (K) e values (V) para uma dimensão menor k << N antes do cálculo de atenção:

**Atenção Padrão:**
```
Attention(Q, K, V) = softmax(QK^T / √d) V
Complexidade: O(N²)
```

**Linformer:**
```
K' = E_k K  (projeta K de N para k)
V' = E_v V  (projeta V de N para k)
Attention(Q, K', V') = softmax(QK'^T / √d) V'
Complexidade: O(N×k)
```

Onde E_k e E_v são matrizes de projeção aprendíveis de dimensão (N, k).

## Arquitetura da Implementação

### 1. LinformerWindowAttention

**Localização**: `utils.py`, linha ~70

**Características**:
- Substitui `WindowAttention` nos blocos Swin Transformer
- Mantém relative position bias (simplificado para compatibilidade)
- Projeta keys e values usando matrizes E_k e E_v de tamanho (window_size², linformer_k)

**Parâmetros**:
- `dim`: Dimensão dos embeddings
- `window_size`: Tamanho da janela (ex: (7, 7))
- `num_heads`: Número de cabeças de atenção
- `linformer_k`: Dimensão de projeção (padrão: 64)

**Exemplo de uso**:
```python
attn = LinformerWindowAttention(
    dim=192,
    window_size=(7, 7),
    num_heads=6,
    linformer_k=64
)
```

### 2. LinformerCrossAttention

**Localização**: `utils.py`, linha ~535

**Características**:
- Substitui `CrossAttention` no módulo DLF
- Suporta sequências de comprimento variável
- Usa `max_seq_len` para inicializar matrizes de projeção
- Trunca ou usa apenas as primeiras N linhas das matrizes E_k e E_v conforme necessário

**Parâmetros**:
- `dim`: Dimensão dos embeddings
- `num_heads`: Número de cabeças de atenção
- `linformer_k`: Dimensão de projeção (padrão: 64)
- `max_seq_len`: Comprimento máximo de sequência (padrão: 512)

**Exemplo de uso**:
```python
attn = LinformerCrossAttention(
    dim=768,
    num_heads=12,
    linformer_k=64,
    max_seq_len=512
)
```

### 3. Integração nos Blocos

#### SwinTransformerBlock

**Modificações**:
- Adicionado parâmetro `use_linformer` e `linformer_k`
- Seleção condicional entre `WindowAttention` e `LinformerWindowAttention`

```python
if use_linformer:
    self.attn = LinformerWindowAttention(...)
else:
    self.attn = WindowAttention(...)
```

#### CrossAttentionBlock

**Modificações**:
- Adicionado parâmetro `use_linformer`, `linformer_k` e `max_seq_len`
- Seleção condicional entre `CrossAttention` e `LinformerCrossAttention`

#### BasicLayer

**Modificações**:
- Propaga parâmetros `use_linformer` e `linformer_k` para todos os `SwinTransformerBlock`

#### MultiScaleBlock

**Modificações**:
- Propaga parâmetros `use_linformer`, `linformer_k` e `max_seq_len` para todos os `CrossAttentionBlock`

### 4. Configuração

**Arquivo**: `configs/BEFUnet_configs.py`

**Novos parâmetros**:
```python
cfg.use_linformer = False  # Habilita/desabilita Linformer
cfg.linformer_k = 64      # Dimensão de projeção
cfg.linformer_max_seq_len = 512  # Para CrossAttention
```

**Integração**:
- `SwinTransformer` recebe `use_linformer` e `linformer_k` via `**kwargs`
- `PyramidFeatures` passa configuração do Linformer para `SwinTransformer`
- `All2Cross` passa configuração do Linformer para `MultiScaleBlock`

## Análise de Complexidade

### Memória

**Atenção Padrão (WindowAttention)**:
- Matriz de atenção: N × N = 49 × 49 = 2,401 elementos por cabeça
- Para 6 cabeças: ~14,406 elementos

**Linformer (LinformerWindowAttention)**:
- Matriz de atenção: N × k = 49 × 64 = 3,136 elementos por cabeça
- Para 6 cabeças: ~18,816 elementos
- **Redução**: ~30% de memória (considerando que k=64 é maior que N=49 para janelas pequenas)

**Para sequências maiores (CrossAttention)**:
- Atenção Padrão: N=3136 → 3136² = 9,834,496 elementos
- Linformer: N=3136, k=64 → 3136×64 = 200,704 elementos
- **Redução**: ~98% de memória!

### Computação

**Atenção Padrão**:
- QK^T: O(N² × d)
- Attention × V: O(N² × d)
- **Total**: O(N² × d)

**Linformer**:
- Projeção K: O(N × k × d)
- Projeção V: O(N × k × d)
- QK'^T: O(N × k × d)
- Attention × V': O(N × k × d)
- **Total**: O(N × k × d)

Para k=64 e N=3136:
- Redução de operações: ~98%

## Uso Prático

### Habilitando Linformer

1. **Via Configuração**:
```python
from configs.BEFUnet_configs import get_BEFUnet_configs

config = get_BEFUnet_configs()
config.use_linformer = True
config.linformer_k = 64
```

2. **Treinamento**:
```bash
python train.py --root_path ./data/Synapse/train_npz \
                 --test_path ./data/Synapse/test_vol_h5 \
                 --model_name BEFUnet \
                 --batch_size 10 \
                 --max_epochs 500
```

### Ajustando Parâmetros

- **`linformer_k`**: Valores menores (32, 48) reduzem mais memória mas podem afetar performance
- **`linformer_k`**: Valores maiores (128, 256) mantêm mais informação mas usam mais memória
- **Recomendação**: Começar com k=64 e ajustar conforme necessário

## Estudos de Ablação Sugeridos

Conforme mencionado nos comentários do código, sugere-se realizar ablações comparando:

1. **BEFUnet baseline** vs **+Linformer**
2. **BEFUnet +depthwise** vs **+Linformer**
3. **BEFUnet +atenção eficiente (Linformer)** vs **+loss (focal-tversky)**
4. **Combinações**: Linformer + Focal-Tversky Loss + Boundary Loss

## Limitações e Considerações

1. **Janelas Pequenas**: Para window_size muito pequeno (ex: 4×4), o Linformer pode não oferecer vantagem significativa
2. **Inicialização**: As matrizes E_k e E_v são inicializadas aleatoriamente; pode ser necessário ajustar a inicialização
3. **Relative Position Bias**: A implementação atual simplifica o relative position bias para compatibilidade com Linformer
4. **Pesos Pré-treinados**: Quando `use_linformer=True`, os pesos de atenção do checkpoint pré-treinado do Swin Transformer não podem ser carregados diretamente, pois a estrutura da atenção muda. O modelo usa `strict=False` para carregar apenas pesos compatíveis (norm, MLP, etc.) e inicializa aleatoriamente os parâmetros de atenção e as matrizes de projeção do Linformer. Isso significa que o treinamento começará do zero para os módulos de atenção quando Linformer estiver habilitado.
5. **Máscara de Atenção**: A aplicação de máscaras (usadas em shifted window attention) foi adaptada para funcionar com a estrutura linear do Linformer, usando média da máscara ao longo da dimensão de keys.

## Referências

1. **Linformer Paper**: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
   - Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020)

2. **BEFUnet Paper**: [BEFUnet: A Hybrid CNN-Transformer Architecture](https://arxiv.org/abs/2402.08793)
   - Manzari, O. N., et al. (2024)

3. **Swin Transformer**: [Swin Transformer: Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)
   - Liu, Z., et al. (2021)

## Contribuições

Esta implementação foi desenvolvida para:
- Reduzir uso de memória durante treinamento
- Acelerar processamento de imagens médicas de alta resolução
- Facilitar estudos de ablação sobre eficiência de atenção
- Fornecer alternativa funcional quando Flash Attention não está disponível

## Notas Técnicas

- A implementação mantém compatibilidade total com o código existente
- Quando `use_linformer=False`, o comportamento é idêntico à versão original
- As matrizes de projeção E_k e E_v são aprendíveis e otimizadas durante o treinamento
- A implementação suporta tanto atenção dentro de janelas quanto cross-attention entre escalas
- **Carregamento de Pesos**: Em `models/Encoder.py`, quando `use_linformer=True`, o código usa `load_state_dict(checkpoint, strict=False)` para permitir que pesos incompatíveis sejam ignorados. Os pesos de atenção (qkv, proj) e as matrizes de projeção do Linformer (E_k, E_v) são inicializados aleatoriamente.
- **Aplicação de Máscaras**: A máscara de atenção (usada em shifted window attention) é adaptada para Linformer calculando a média ao longo da dimensão de keys, resultando em uma máscara de shape `(nW, N)` que é expandida para fazer broadcast com a atenção de shape `(B_ // nW, nW, num_heads, N, linformer_k)`.

