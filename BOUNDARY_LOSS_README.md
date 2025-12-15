# Implementação do Boundary Loss para BEFUnet

## Visão Geral

Este documento descreve a implementação do **Boundary Loss** no projeto BEFUnet, conforme sugerido pelo professor Cleber Zanchettin. O Boundary Loss é uma função de perda projetada para melhorar a segmentação de classes minoritárias em problemas altamente desbalanceados, com foco especial na precisão das bordas das segmentações.

## Motivação

Em segmentação de imagens médicas, a precisão das bordas é crítica para diagnósticos precisos. O Boundary Loss foi proposto para lidar com problemas onde:

1. **Classes minoritárias** têm áreas significativamente menores que classes majoritárias
2. **Precisão de bordas** é mais importante que precisão de região
3. **Segmentação desbalanceada** onde métricas regionais (como Dice) podem não ser suficientes

## Referência Científica

**Paper Original:** Kervadec et al. "Boundary loss for highly unbalanced segmentation"  
**Publicação:** Nature Methods (2020)  
**DOI:** https://www.nature.com/articles/s41592-020-01008-z

## Implementação Técnica

### Classe BoundaryLoss

A implementação está localizada em `utils.py` e consiste em:

1. **`_one_hot_encoder`**: Converte índices de classe em codificação one-hot
2. **`_compute_distance_map`**: Calcula o mapa de distância usando transformada de distância euclidiana
3. **`forward`**: Computa a perda de boundary multiplicando as predições pelo mapa de distância

### Como Funciona

O Boundary Loss funciona através dos seguintes passos:

1. **Cálculo do Mapa de Distância**: Para cada classe na ground truth, calcula-se um mapa de distância onde cada pixel contém a distância até a borda mais próxima da região segmentada.

2. **Aplicação da Perda**: As probabilidades preditas são multiplicadas pelo mapa de distância. Isso penaliza:
   - Predições dentro da região mas longe da borda verdadeira
   - Predições fora da região mas próximas à borda verdadeira

3. **Normalização**: A perda é normalizada pelo tamanho do batch e número de classes.

### Fórmula Matemática

Para uma classe `c`, o boundary loss é calculado como:

```
L_boundary = Σ(pred_c * dist_map_c) / (B * C)
```

Onde:
- `pred_c`: Probabilidades preditas para a classe `c`
- `dist_map_c`: Mapa de distância da ground truth da classe `c`
- `B`: Tamanho do batch
- `C`: Número de classes

## Integração no Treinamento

### Combinação com Outras Losses

O Boundary Loss é combinado com Cross Entropy e Dice Loss:

```python
loss_total = weight_ce * loss_ce + weight_dice * loss_dice + weight_boundary * loss_boundary
```

**Pesos Padrão:**
- Cross Entropy: 0.4
- Dice Loss: 0.5
- Boundary Loss: 0.1

### Configuração

Os pesos podem ser configurados via argumentos de linha de comando:

```bash
python train.py \
    --weight_ce 0.4 \
    --weight_dice 0.5 \
    --weight_boundary 0.1 \
    --root_path ./data/Synapse/train_npz \
    --test_path ./data/Synapse/test_vol_h5 \
    --model_name BEFUnet \
    --batch_size 10 \
    --max_epochs 500
```

### Monitoramento

Durante o treinamento, as seguintes métricas são registradas no TensorBoard:

- `info/total_loss`: Perda total combinada
- `info/loss_ce`: Cross Entropy Loss
- `info/loss_dice`: Dice Loss
- `info/loss_boundary`: Boundary Loss

## Uso no Google Colab

Para usar esta implementação no Google Colab:

1. **Clone o repositório** (ou faça upload dos arquivos):
```python
!git clone <repository_url>
%cd BEFUnet
```

2. **Instale as dependências**:
```python
!pip install -r requirements.txt
```

3. **Execute o treinamento** com Boundary Loss:
```python
!python train.py \
    --root_path ./data/Synapse/train_npz \
    --test_path ./data/Synapse/test_vol_h5 \
    --model_name BEFUnet \
    --batch_size 10 \
    --eval_interval 20 \
    --max_epochs 500 \
    --weight_boundary 0.1
```

4. **Monitore o progresso** através dos logs e TensorBoard:
```python
# Visualizar logs
!tail -f results/BEFUnet/*_log.txt

# TensorBoard (se disponível)
%load_ext tensorboard
%tensorboard --logdir results/BEFUnet/log
```

## Ablations Sugeridas

Conforme sugerido pelo professor, as seguintes ablations podem ser realizadas:

1. **BEFUnet baseline** vs **+depthwise** vs **+atenção eficiente** vs **+loss (focal-tversky)** vs **+boundary loss** vs **combinação**

### Exemplo de Ablation para Boundary Loss

Para treinar apenas com Boundary Loss (sem outras losses):

```bash
python train.py \
    --weight_ce 0.0 \
    --weight_dice 0.0 \
    --weight_boundary 1.0 \
    --model_name BEFUnet_boundary_only
```

Para combinação padrão (CE + Dice + Boundary):

```bash
python train.py \
    --weight_ce 0.4 \
    --weight_dice 0.5 \
    --weight_boundary 0.1 \
    --model_name BEFUnet_combined
```

## Estrutura de Arquivos Modificados

- **`utils.py`**: Adicionada classe `BoundaryLoss`
- **`trainer.py`**: Integração do Boundary Loss no loop de treinamento
- **`train.py`**: Adicionados argumentos para configuração dos pesos das losses

## Dependências Adicionais

A implementação utiliza:
- `scipy.ndimage.distance_transform_edt`: Para cálculo do mapa de distância euclidiana

Esta dependência já está incluída no `requirements.txt` padrão do projeto.

## Observações Importantes

1. **Performance**: O cálculo do mapa de distância é computacionalmente mais custoso que Dice ou CE. Para batches grandes, pode haver um impacto no tempo de treinamento.

2. **Memória**: O mapa de distância é calculado na CPU e depois transferido para GPU. Para imagens muito grandes, considere ajustar o batch size.

3. **Hiperparâmetros**: Os pesos das losses devem ser ajustados conforme o dataset e problema específico. Valores sugeridos:
   - Para problemas com classes muito desbalanceadas: aumentar `weight_boundary` (0.2-0.3)
   - Para problemas com bordas críticas: aumentar `weight_boundary` (0.15-0.25)

## Resultados Esperados

Com o Boundary Loss, espera-se:

- **Melhoria na precisão de bordas** (métrica HD95)
- **Melhor segmentação de classes minoritárias**
- **Redução de falsos positivos próximos às bordas**

## Troubleshooting

### Erro: "CUDA out of memory"
- Reduza o `batch_size`
- O Boundary Loss usa mais memória devido ao cálculo do mapa de distância

### Boundary Loss muito alto ou muito baixo
- Ajuste o peso `weight_boundary` conforme necessário
- Verifique se as predições estão no formato correto (probabilidades após softmax)

### Performance lenta
- O cálculo do mapa de distância é feito na CPU. Para acelerar, considere pré-computar os mapas de distância ou usar implementação GPU (se disponível)

