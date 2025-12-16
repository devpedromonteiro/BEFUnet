# Combinação de Melhorias no BEFUnet

## Visão Geral

Este documento descreve a **branch `combination`**, que integra todas as 4 melhorias propostas pelo professor Cleber Zanchettin para aprimorar o BEFUnet. Esta combinação representa a versão final do modelo com todas as otimizações de arquitetura e função de perda trabalhando em conjunto.

## Por que Combinar as 4 Melhorias?

As 4 melhorias foram desenvolvidas em branches separadas para permitir análise individual (ablation study). No entanto, o professor sugeriu testar a **combinação de todas elas simultaneamente**, pois:

1. **Sinergia entre melhorias**: Cada melhoria aborda um aspecto diferente do modelo:
   - **Depthwise Separable Convolutions**: Reduz parâmetros e MACs (eficiência)
   - **FlashAttention-2**: Acelera atenção e reduz memória (eficiência)
   - **Focal Tversky Loss**: Melhora classes minoritárias (qualidade)
   - **Boundary Loss**: Melhora precisão de bordas (qualidade)

2. **Complementaridade**: As melhorias não conflitam entre si:
   - Otimizações de arquitetura (depthwise, FlashAttention) são independentes
   - Funções de perda podem ser combinadas com pesos configuráveis

3. **Avaliação completa**: A combinação permite avaliar o impacto conjunto de todas as melhorias, que é o objetivo final do projeto de pesquisa.

## As 4 Melhorias Combinadas

### 1. Depthwise Separable Convolutions

**O que faz**: Substitui convoluções padrão 3×3 por convoluções depthwise separable no decoder, reduzindo significativamente parâmetros e operações computacionais.

**Onde**: `models/Decoder.py` (ConvUpsample e SegmentationHead)

**Benefícios**:
- Redução de ~75% nos parâmetros das convoluções 3×3
- Redução de ~67% nos MACs
- Mantém performance similar com muito menos recursos

**Documentação**: Ver [DEPTHWISE_SEPARABLE_README.md](DEPTHWISE_SEPARABLE_README.md)

### 2. FlashAttention-2

**O que faz**: Implementa atenção eficiente usando FlashAttention-2 no módulo de atenção do encoder, acelerando o cálculo e reduzindo uso de memória GPU.

**Onde**: `models/Encoder.py` (classe `Attention`)

**Benefícios**:
- 2-4x mais rápido que atenção padrão
- Redução significativa no uso de memória GPU
- Fallback automático se FlashAttention-2 não estiver instalado

**Documentação**: Ver [FLASH_ATTENTION_README.md](FLASH_ATTENTION_README.md)

### 3. Focal Tversky Loss

**O que faz**: Substitui parcialmente Cross-Entropy Loss por Focal Tversky Loss, que é mais eficaz para classes minoritárias através de foco em exemplos difíceis.

**Onde**: `utils.py` (classe `FocalTverskyLoss`)

**Benefícios**:
- Melhor performance em classes minoritárias
- Foco em exemplos difíceis através do parâmetro gamma
- Controle de falsos positivos/negativos via alpha/beta

**Documentação**: Ver [FOCAL_TVERSKY_LOSS.md](FOCAL_TVERSKY_LOSS.md)

### 4. Boundary Loss

**O que faz**: Adiciona uma função de perda que penaliza predições distantes das bordas verdadeiras, melhorando a precisão de contornos em segmentação médica.

**Onde**: `utils.py` (classe `BoundaryLoss`)

**Benefícios**:
- Melhora precisão de bordas (crítico para diagnóstico médico)
- Complementa métricas regionais como Dice
- Eficaz para segmentação altamente desbalanceada

**Documentação**: Ver [BOUNDARY_LOSS_README.md](BOUNDARY_LOSS_README.md)

## Como as Melhorias Foram Combinadas

### Estratégia de Merge

A combinação foi realizada através de **merge sequencial** das branches na seguinte ordem:

1. **baseline** → **depthwise** (sem conflitos)
2. **combination** → **efficient-attention** (conflito menor no notebook)
3. **combination** → **focal-tversky-loss** (mantendo CrossEntropyLoss)
4. **combination** → **boundary-loss** (resolvendo conflitos em utils.py, trainer.py, train.py)

### Resolução de Conflitos

#### `utils.py`
- **Conflito**: Ambas as branches `focal-tversky-loss` e `boundary-loss` adicionavam classes de loss diferentes
- **Solução**: Mantidas ambas as classes `FocalTverskyLoss` e `BoundaryLoss` lado a lado

#### `trainer.py`
- **Conflito**: 
  - `focal-tversky-loss` substituía CrossEntropyLoss por FocalTverskyLoss
  - `boundary-loss` adicionava BoundaryLoss junto com CE e Dice
- **Solução**: 
  - Mantido CrossEntropyLoss (não removido)
  - Adicionadas todas as 4 losses: `ce_loss`, `dice_loss`, `focal_tversky_loss`, `boundary_loss`
  - Combinadas com pesos configuráveis

#### `train.py`
- **Conflito**: Apenas `boundary-loss` tinha argumentos de peso
- **Solução**: Adicionados argumentos de peso para todas as 4 losses

## Estrutura Final

### Funções de Perda Combinadas

A loss final é uma combinação ponderada de todas as 4 losses:

```python
loss = (weight_ce * loss_ce + 
        weight_dice * loss_dice + 
        weight_focal_tversky * loss_focal_tversky + 
        weight_boundary * loss_boundary)
```

**Pesos padrão**:
- `weight_ce = 0.2` (Cross Entropy)
- `weight_dice = 0.3` (Dice Loss)
- `weight_focal_tversky = 0.3` (Focal Tversky Loss)
- `weight_boundary = 0.2` (Boundary Loss)

Os pesos são normalizados automaticamente para somar 1.0.

### Arquivos Modificados

#### Arquivos de Código Principal
- **`models/Decoder.py`**: Depthwise separable convolutions
- **`models/Encoder.py`**: FlashAttention-2 support
- **`utils.py`**: Classes `FocalTverskyLoss` e `BoundaryLoss` adicionadas
- **`trainer.py`**: Combinação de todas as 4 losses
- **`train.py`**: Argumentos de peso para todas as losses

#### Arquivos Novos
- **`models/depthwise_separable.py`**: Implementação de depthwise separable convolution
- **`test_depthwise_separable.py`**: Testes para depthwise separable
- **`test_flash_attention.py`**: Testes para FlashAttention-2

#### Documentação
- **`DEPTHWISE_SEPARABLE_README.md`**: Documentação depthwise
- **`FLASH_ATTENTION_README.md`**: Documentação FlashAttention-2
- **`FOCAL_TVERSKY_LOSS.md`**: Documentação Focal Tversky Loss
- **`BOUNDARY_LOSS_README.md`**: Documentação Boundary Loss
- **`COMBINATION_README.md`**: Este arquivo

## Como Usar

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt

# Opcional: Instalar FlashAttention-2 para aceleração
pip install flash-attn --no-build-isolation
```

**Nota**: Se FlashAttention-2 não estiver instalado, o modelo usa atenção padrão automaticamente (fallback).

### 2. Treinamento com Combinação Completa

```bash
python train.py \
    --root_path ./data/Synapse/train_npz \
    --test_path ./data/Synapse/test_vol_h5 \
    --model_name BEFUnet \
    --batch_size 10 \
    --eval_interval 20 \
    --max_epochs 500 \
    --weight_ce 0.2 \
    --weight_dice 0.3 \
    --weight_focal_tversky 0.3 \
    --weight_boundary 0.2
```

### 3. Ajustar Pesos das Losses

Você pode ajustar os pesos conforme necessário:

```bash
# Exemplo: Dar mais peso para Boundary Loss (melhorar bordas)
python train.py ... \
    --weight_ce 0.15 \
    --weight_dice 0.25 \
    --weight_focal_tversky 0.25 \
    --weight_boundary 0.35

# Exemplo: Focar em classes minoritárias (mais Focal Tversky)
python train.py ... \
    --weight_ce 0.1 \
    --weight_dice 0.3 \
    --weight_focal_tversky 0.5 \
    --weight_boundary 0.1
```

**Importante**: Os pesos são normalizados automaticamente, então você pode usar valores proporcionais.

### 4. Monitoramento durante Treinamento

O TensorBoard registra todas as losses individualmente:

```bash
tensorboard --logdir ./results/BEFUnet/log
```

Você verá:
- `info/total_loss`: Loss combinada total
- `info/loss_ce`: Cross Entropy Loss
- `info/loss_dice`: Dice Loss
- `info/loss_focal_tversky`: Focal Tversky Loss
- `info/loss_boundary`: Boundary Loss

## Benefícios Esperados da Combinação

### Eficiência
- **Menos parâmetros**: Depthwise separable reduz parâmetros do decoder
- **Menos memória**: FlashAttention-2 reduz uso de memória GPU
- **Treinamento mais rápido**: Ambas as otimizações aceleram o processo

### Qualidade
- **Melhor em classes minoritárias**: Focal Tversky Loss foca em exemplos difíceis
- **Melhor precisão de bordas**: Boundary Loss penaliza erros de contorno
- **Métricas regionais e de borda**: Combinação de Dice (região) e Boundary (borda)

### Robustez
- **Fallback automático**: Se FlashAttention-2 não estiver disponível, usa atenção padrão
- **Pesos configuráveis**: Permite ajuste fino para diferentes datasets/tarefas
- **Compatibilidade**: Mantém compatibilidade com código original

## Ablation Study

Para comparar o impacto de cada melhoria, você pode usar as branches individuais:

- **`baseline`**: Versão original do BEFUnet
- **`depthwise`**: Baseline + Depthwise Separable Convolutions
- **`efficient-attention`**: Baseline + FlashAttention-2
- **`focal-tversky-loss`**: Baseline + Focal Tversky Loss
- **`boundary-loss`**: Baseline + Boundary Loss
- **`combination`**: Todas as melhorias combinadas (esta branch)

## Resultados Esperados

Com a combinação completa, espera-se:

1. **Redução de parâmetros**: ~20-30% menos parâmetros no decoder
2. **Redução de memória**: ~30-50% menos memória GPU durante treinamento
3. **Melhoria em Dice/IoU**: Especialmente para classes minoritárias
4. **Melhoria em HD95**: Melhor precisão de bordas (métrica sugerida pelo professor)
5. **Tempo de treinamento**: Redução de ~20-40% devido às otimizações

## Observações Importantes

### Sobre o SegPC 2021 Dataset

O professor mencionou que o SegPC 2021 (mieloma múltiplo, histopatologia) tem:
- Anotações desafiadoras
- Ruído de rótulos reportado nas submissões vencedoras
- Necessidade de pré-processamento e possível correção/pseudorrótulos de borda

A combinação de Boundary Loss + Focal Tversky Loss é especialmente adequada para este tipo de dataset desafiador.

### Métricas de Avaliação

Conforme sugerido pelo professor, além de Dice/IoU, é importante usar:
- **HD95** (Hausdorff Distance 95th percentile): Já implementado
- **Average Surface Distance**: Pode ser adicionado se necessário

Essas métricas capturam melhor erros de contorno que Dice/IoU não capturam bem.

## Referências

1. **Depthwise Separable Convolutions**: 
   - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
   - https://arxiv.org/abs/1704.04861

2. **FlashAttention-2**:
   - FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
   - https://github.com/Dao-AILab/flash-attention

3. **Focal Tversky Loss**:
   - Abraham, N., & Khan, N. M. (2019). A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation
   - https://www.nature.com/articles/s41592-020-01008-z

4. **Boundary Loss**:
   - Kervadec et al. "Boundary loss for highly unbalanced segmentation"
   - Nature Methods (2020)
   - https://www.nature.com/articles/s41592-020-01008-z

## Autor

João Pedro Monteiro Pereira - jpmp2@cin.ufpe.br

## Data

Dezembro 2025

