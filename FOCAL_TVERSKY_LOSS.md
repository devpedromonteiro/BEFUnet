# Implementação da Focal-Tversky Loss para Classes Minoritárias

## 📋 Visão Geral

Este documento descreve a implementação da **Focal-Tversky Loss** como substituição da Cross-Entropy Loss pura no modelo BEFUnet, conforme sugestão do professor para melhorar o desempenho em classes minoritárias.

## 🎯 Objetivo

A implementação foi realizada para atender à sugestão de substituir a Cross-Entropy pura por **Focal-Tversky Loss** (ou Dice Focal) para classes minoritárias, conforme referência do artigo:
- **Paper**: [Nature Methods - Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://www.nature.com/articles/s41592-020-01008-z)

## 🔬 Contexto da Ablação

Esta implementação faz parte de uma série de ablações sugeridas:
1. BEFUnet baseline
2. +depthwise
3. +atenção eficiente
4. **+loss (focal-tversky)** ← **Esta implementação**
5. +boundary loss
6. Combinações

## 📝 O que foi Implementado

### 1. Nova Classe: `FocalTverskyLoss`

**Localização**: `utils.py`

Uma nova classe de loss function foi criada seguindo o padrão do `DiceLoss` existente:

```python
class FocalTverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6):
        ...
```

#### Parâmetros:

- **`n_classes`**: Número de classes de segmentação
- **`alpha`** (default: 0.3): Peso para falsos positivos (FP)
- **`beta`** (default: 0.7): Peso para falsos negativos (FN) - valores maiores focam em recall
- **`gamma`** (default: 4/3): Parâmetro focal para focar em exemplos difíceis
- **`smooth`** (default: 1e-6): Fator de suavização para evitar divisão por zero

#### Características:

- ✅ Suporte para segmentação **multi-classe**
- ✅ Compatível com o formato de entrada do modelo (logits ou probabilidades)
- ✅ Usa **one-hot encoding** para targets (mesmo padrão do DiceLoss)
- ✅ Calcula a loss para cada classe e retorna a média ponderada

### 2. Modificações no `trainer.py`

#### Substituição da Loss:

**Antes:**
```python
ce_loss = CrossEntropyLoss()
loss_ce = ce_loss(outputs, label_batch[:].long())
loss = 0.4 * loss_ce + 0.6 * loss_dice
```

**Depois:**
```python
focal_tversky_loss = FocalTverskyLoss(n_classes=num_classes, alpha=0.3, beta=0.7, gamma=4/3)
loss_ft = focal_tversky_loss(outputs, label_batch[:].long(), softmax=True)
loss = 0.4 * loss_ft + 0.6 * loss_dice
```

#### Mudanças nos Logs:

- Logs atualizados para registrar `loss_focal_tversky` em vez de `loss_ce`
- TensorBoard agora mostra `info/loss_focal_tversky` ao invés de `info/loss_ce`

## 🧮 Como Funciona a Focal-Tversky Loss

### Fórmula Matemática:

A Focal-Tversky Loss combina dois conceitos:

1. **Tversky Index (TI)**:
   ```
   TI = TP / (TP + α·FP + β·FN)
   ```
   Onde:
   - TP = True Positives
   - FP = False Positives  
   - FN = False Negatives
   - α e β são pesos que controlam a importância de FP e FN

2. **Focal Component**:
   ```
   Focal-Tversky Loss = (1 - TI)^γ
   ```
   Onde γ é o parâmetro focal que aumenta o foco em exemplos difíceis.

### Por que é Eficaz para Classes Minoritárias?

1. **β > α**: Com `beta=0.7` e `alpha=0.3`, a loss penaliza mais os falsos negativos, melhorando o **recall** para classes minoritárias.

2. **Focal Effect**: O parâmetro `gamma` aumenta o peso de exemplos difíceis de classificar, que são comuns em classes minoritárias.

3. **Combinação com Dice**: A combinação `0.4 * FocalTversky + 0.6 * Dice` mantém o benefício do Dice Loss enquanto adiciona o foco em classes minoritárias.

## 🚀 Como Usar

### Treinamento

O código está pronto para uso. Basta executar o treinamento normalmente:

```bash
python train.py --root_path ./data/Synapse/train_npz \
                --test_path ./data/Synapse/test_vol_h5 \
                --model_name BEFUnet \
                --batch_size 10 \
                --eval_interval 20 \
                --max_epochs 500
```

A Focal-Tversky Loss será usada automaticamente no lugar da Cross-Entropy.

### Ajuste de Parâmetros

Se desejar ajustar os parâmetros da Focal-Tversky Loss, edite o arquivo `trainer.py` na linha 102:

```python
focal_tversky_loss = FocalTverskyLoss(
    n_classes=num_classes, 
    alpha=0.3,    # Ajuste para controlar peso de FP
    beta=0.7,     # Ajuste para controlar peso de FN (maior = mais foco em recall)
    gamma=4/3     # Ajuste para controlar foco em exemplos difíceis
)
```

**Recomendações**:
- Para classes muito minoritárias: aumentar `beta` (ex: 0.8) e `gamma` (ex: 1.5)
- Para balancear precisão e recall: `alpha=0.5`, `beta=0.5`
- Para foco máximo em recall: `beta=0.9`, `alpha=0.1`

### Ajuste da Combinação de Losses

Para alterar a proporção entre Focal-Tversky e Dice, edite a linha 175 do `trainer.py`:

```python
loss = 0.4 * loss_ft + 0.6 * loss_dice  # Proporção atual
# Exemplos alternativos:
# loss = 0.5 * loss_ft + 0.5 * loss_dice  # Balanceado
# loss = 0.3 * loss_ft + 0.7 * loss_dice  # Mais foco no Dice
```

## 📊 Monitoramento

### TensorBoard

Durante o treinamento, você pode monitorar:

- `info/loss_focal_tversky`: Valor da Focal-Tversky Loss
- `info/loss_dice`: Valor da Dice Loss
- `info/total_loss`: Loss total combinada

### Logs de Treinamento

Os logs agora mostram:
```
iteration X : loss : Y, loss_focal_tversky: Z loss_dice: W
```

## 🔍 Verificação da Implementação

### Arquivos Modificados:

1. **`utils.py`**:
   - Adicionada classe `FocalTverskyLoss` (linhas ~590-670)

2. **`trainer.py`**:
   - Import atualizado para incluir `FocalTverskyLoss` (linha 13)
   - Substituição de `CrossEntropyLoss` por `FocalTverskyLoss` (linha 102)
   - Atualização do cálculo de loss (linha 173)
   - Atualização dos logs (linhas 187-189)

### Compatibilidade:

- ✅ Compatível com o código existente
- ✅ Mantém a mesma interface do DiceLoss
- ✅ Não requer mudanças no dataset ou modelo
- ✅ Pronto para uso no Google Colab

## 📚 Referências

1. **Paper Principal**:
   - Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. In *International workshop on machine learning in medical imaging* (pp. 379-387). Springer.
   - Nature Methods: https://www.nature.com/articles/s41592-020-01008-z

2. **Focal Loss Original**:
   - Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).

## 🎓 Notas para o Colab

Ao executar no Google Colab:

1. Certifique-se de que todos os arquivos estão no ambiente
2. A implementação não requer dependências adicionais
3. Os logs mostrarão `loss_focal_tversky` em vez de `loss_ce`
4. O treinamento seguirá o mesmo fluxo, mas com melhor foco em classes minoritárias

## ✅ Checklist de Implementação

- [x] Classe `FocalTverskyLoss` implementada
- [x] Integração no `trainer.py` concluída
- [x] Logs e métricas atualizados
- [x] Compatibilidade mantida com código existente
- [x] Documentação criada
- [x] Pronto para execução no Colab

---

**Data de Implementação**: Dezembro 2025 
**Versão**: 1.0  
**Status**: ✅ Implementação Completa e Testada

