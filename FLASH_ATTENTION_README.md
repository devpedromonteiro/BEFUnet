# FlashAttention-2 Implementation

## Resumo

Este documento descreve a implementação de **FlashAttention-2** no BEFUnet para acelerar o cálculo de atenção e reduzir o uso de memória GPU.

## O que é FlashAttention-2?

FlashAttention-2 é uma implementação otimizada do mecanismo de atenção que:
- **Reduz memória**: Usa técnicas de tiling e recomputação para evitar armazenar a matriz de atenção completa
- **Acelera computação**: 2-4x mais rápido que atenção padrão
- **Mantém precisão**: Resultados numericamente equivalentes (com pequenas diferenças devido a recomputação)

## Onde foi implementado

### 1. Classe `Attention` (models/Encoder.py)
- ✅ **Implementado com FlashAttention-2**
- Usado no módulo LCAF (Local Cross-Attention Fusion)
- Formato compatível: `(batch, seq_len, num_heads, head_dim)`
- Fallback automático para atenção padrão se FlashAttention-2 não estiver disponível

### 2. Classe `WindowAttention` (utils.py)
- ⚠️ **Não implementado** (mais complexo)
- Razão: Usa `relative_position_bias` que precisa ser adicionado antes do softmax
- FlashAttention-2 faz softmax internamente, dificultando a integração do bias
- Possível solução futura: Adicionar bias após FlashAttention ou usar versão modificada

## Instalação

### Opção 1: Instalação via pip (recomendado)
```bash
pip install flash-attn --no-build-isolation
```

### Opção 2: Instalação do código fonte (se pip falhar)
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install .
```

### Requisitos
- CUDA 11.6 ou superior
- PyTorch 1.12 ou superior
- GPU compatível (NVIDIA com compute capability >= 7.0)

## Como usar

### Uso padrão (com FlashAttention-2 se disponível)
```python
from models.Encoder import Attention

# FlashAttention-2 será usado automaticamente se disponível
attn = Attention(dim=512, factor=1, heads=8, dim_head=64, dropout=0.1)
```

### Desabilitar FlashAttention-2 (forçar atenção padrão)
```python
# Se quiser usar atenção padrão mesmo com FlashAttention-2 instalado
attn = Attention(dim=512, factor=1, heads=8, dim_head=64, dropout=0.1, 
                 use_flash_attention=False)
```

## Compatibilidade

- ✅ **Totalmente compatível** com código existente
- ✅ **Fallback automático**: Se FlashAttention-2 não estiver disponível, usa atenção padrão
- ✅ **Mesma interface**: Não requer mudanças no código que usa a classe
- ✅ **Mesmo comportamento**: Output shapes e forward pass idênticos

## Verificação

O código verifica automaticamente se FlashAttention-2 está disponível:

```python
# Se FlashAttention-2 não estiver instalado, você verá:
# "FlashAttention-2 not available. Using standard attention. Install with: pip install flash-attn"
```

## Benefícios Esperados

### Performance
- **Velocidade**: 2-4x mais rápido em GPUs modernas (A100, H100)
- **Memória**: Redução significativa no uso de memória GPU
- **Escalabilidade**: Melhor para sequências longas

### Limitações
- Requer GPU NVIDIA compatível
- Pode não funcionar em GPUs antigas (compute capability < 7.0)
- Instalação pode ser complicada em alguns ambientes

## Troubleshooting

### Erro: "FlashAttention-2 not available"
- **Solução**: Instale com `pip install flash-attn --no-build-isolation`
- Se ainda falhar, instale do código fonte (veja Instalação)

### Erro: "CUDA out of memory" mesmo com FlashAttention-2
- FlashAttention-2 reduz memória, mas não elimina completamente o problema
- Tente reduzir batch size ou sequence length

### Performance não melhorou
- Verifique se está usando GPU (não CPU)
- Verifique se FlashAttention-2 está realmente sendo usado (veja mensagem de import)
- Algumas GPUs antigas podem não ter ganho significativo

## Referências

- Paper original: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- Repositório: https://github.com/Dao-AILab/flash-attention
- Documentação: https://github.com/Dao-AILab/flash-attention/blob/main/README.md

## Próximos Passos

1. Testar no ambiente de treinamento (Google Colab)
2. Comparar velocidade e uso de memória com/sem FlashAttention-2
3. Verificar se métricas (DSC, HD95) permanecem similares
4. Documentar resultados no artigo

## Notas Técnicas

- FlashAttention-2 usa recomputação para economizar memória, então resultados podem ter pequenas diferenças numéricas
- O dropout é aplicado internamente pelo FlashAttention-2
- A implementação atual funciona apenas para atenção bidirecional (não causal)


