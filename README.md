# 🇧🇷 Detector de Autenticidade de Cédulas BRL

Sistema de visão computacional em Python que classifica cédulas do Real Brasileiro como **Verdadeiras** ou **Falsas** em tempo real via câmera, usando múltiplas técnicas de análise de imagem e aprendizado de máquina.

---

## Estrutura do projeto

```
├── preprocessamento.py   # 9 camadas de transformação de imagem
├── classificador.py      # 10 técnicas de análise + treinamento SVM
├── comunicacao.py        # Console colorido, CSV, JSON, estatísticas
├── main.py               # Câmera em tempo real, threads, CLI
├── testes.py             # Suite de testes unitários
├── modelos/              # Criada automaticamente
│   ├── svm_notas.pkl     # Modelo SVM treinado (gerado após treino)
│   └── template_orb.pkl  # Keypoints ORB de referência
└── dataset/              # Criada pelo usuário para treino
    ├── verdadeiras/       # Fotos de notas autênticas
    └── falsas/            # Fotos de notas falsas
```

---

## Instalação

### Dependências obrigatórias
```bash
pip install cv2 opencv-python numpy
```

### Dependências opcionais (ativam técnicas adicionais)
```bash
pip install scikit-image          # LBP, entropia, HOG
pip install scikit-learn joblib   # SVM treinável
pip install face_recognition      # Verificação de retrato na cédula
```

> **Nota:** `face_recognition` requer `cmake` e `dlib`. Em sistemas Debian/Ubuntu:
> ```bash
> sudo apt install cmake build-essential
> pip install dlib face_recognition
> ```

---

## Uso rápido

### Câmera em tempo real
```bash
python main.py
```

### Escolher câmera e modo de análise
```bash
python main.py --cameras 0 1 --modo SOBEL
```

### Analisar uma imagem estática
```bash
python main.py --imagem foto_nota.jpg --modo COMBINADO
```

### Gravar log dos resultados
```bash
python main.py --csv resultados.csv --json snapshot.json
```

### Abrir painel de diagnóstico ao iniciar
```bash
python main.py --diagnostico
```

### Ver todas as opções
```bash
python main.py --help
```
## Fluxo de treinamento

### 1. Treino offline com pasta de imagens
```bash
python classificador.py --treinar \
  --verdadeiras dataset/verdadeiras/ \
  --falsas dataset/falsas/
```

### 2. Salvar template ORB de uma nota de referência
```bash
python classificador.py --template nota_real.jpg
```

### 3. Treino em tempo real durante uso
```bash
python main.py   # pressione v=verdadeira  f=falsa  t=treinar
```

---

## Teclas durante a execução

| Tecla | Ação |
|-------|------|
| `q` | Encerra a câmera |
| `m` | Alterna para o próximo modo de análise |
| `s` | Salva screenshot do frame anotado |
| `d` | Abre/fecha o painel de diagnóstico (9 camadas) |
| `v` | Coleta o frame atual como amostra **Verdadeira** (treino) |
| `f` | Coleta o frame atual como amostra **Falsa** (treino) |
| `t` | Treina o SVM com as amostras coletadas na sessão |
| `r` | Reseta os contadores de estatísticas |

---

## Modos de análise (`--modo`)

| Modo | Técnica | O que analisa |
|------|---------|---------------|
| `COMBINADO` | Fusão ponderada | Resultado de todos os métodos abaixo (padrão) |
| `GRAY` | Estatísticas de cinza | Variância e entropia de luminância |
| `SOBEL` | Filtro Sobel | Densidade e uniformidade de bordas (calcografia) |
| `LAPLACE` | Filtro Laplaciano | Nitidez de microimpressões e traços finos |
| `FFT` | Transformada de Fourier | Padrões periódicos do guilhoché |
| `LBP` | Local Binary Pattern | Microestrutura e textura do papel-moeda |
| `HIST_COR` | Histograma HSV | Elementos cromáticos de segurança (OVI, holograma) |
| `ORB` | Keypoints ORB | Correspondência com template de nota autêntica |
| `HOG` | Gradientes orientados | Estrutura global de gradientes da cédula |
| `FACE` | face_recognition | Detecção do retrato na cédula |
| `SVM` | Máquina de vetores | Modelo treinado com amostras reais |

Os pesos da fusão no modo `COMBINADO` são:

```
GRAY=10%  SOBEL=15%  LAPLACE=15%  FFT=15%  LBP=10%
HIST_COR=10%  ORB=10%  HOG=10%  FACE=5%
```

---

## Camadas de pré-processamento

Cada frame passa pelas seguintes transformações antes da análise (visíveis com `d`):

| Camada | Descrição |
|--------|-----------|
| `gray` | Escala de cinza |
| `clahe` | Equalização adaptativa (CLAHE) — realça microimpressões |
| `sobel` | Magnitude do gradiente Sobel (H² + V²)^0.5 |
| `laplace` | Filtro Laplaciano — realça variações abruptas de textura |
| `canny` | Mapa de bordas Canny |
| `thresh` | Limiarização adaptativa gaussiana |
| `fft_mag` | Magnitude da FFT — revela padrões de guilhoché |
| `nitidez` | Mapa de nitidez local (variância em janela 7×7) |
| `original` | Frame original normalizado (640 px de largura) |

---

## Treinamento do modelo SVM

O SVM aprende a distinguir notas verdadeiras de falsas a partir de exemplos fornecidos pelo usuário. Quanto mais amostras, maior a precisão.

### Opção 1 — Treino offline com pasta de imagens (recomendado)

```
dataset/
├── verdadeiras/   ← fotos de cédulas autênticas (.jpg / .png)
└── falsas/        ← fotos de cédulas falsas
```

```bash
python classificador.py --treinar \
  --verdadeiras dataset/verdadeiras/ \
  --falsas      dataset/falsas/
```

O modelo é salvo automaticamente em `modelos/svm_notas.pkl` e carregado nas próximas execuções.

### Opção 2 — Treino em tempo real via câmera

1. Execute `python main.py`
2. Posicione uma nota verdadeira na câmera → pressione `v`
3. Posicione uma nota falsa na câmera → pressione `f`
4. Repita até ter ao menos 4 amostras de cada tipo
5. Pressione `t` para treinar

### Salvar template ORB de referência

O modo `ORB` compara keypoints com uma nota-template cadastrada. Para cadastrar:

```bash
python classificador.py --template foto_nota_verdadeira.jpg
```

---

## Saídas

### Terminal (colorido)
```
[14:32:01] cam=0  ✔ VERDADEIRA  87.3%  [COMBINADO]
          GRAY        82.1%
          SOBEL       91.4%
          LAPLACE     78.9%
          FFT         85.0%
          ...
```

### CSV (`--csv log.csv`)
```
timestamp,camera,metodo,status,score,detalhes
2025-06-01T14:32:01.412,0,COMBINADO,Verdadeira,87.30,"{...}"
```

### JSON (`--json snapshot.json`)
```json
{
  "timestamp": "2025-06-01T14:32:01.412",
  "camera": "0",
  "status": "Verdadeira",
  "score": 87.3,
  "metodo": "COMBINADO",
  "detalhes": {
    "GRAY":    { "status": "Verdadeira", "score": 82.1 },
    "SOBEL":   { "status": "Verdadeira", "score": 91.4 }
  }
}
```

---

## Testes

```bash
python testes.py -v
```

A suite cobre pré-processamento, todas as 10 técnicas individualmente, o modo combinado, o treinador SVM e os módulos de comunicação. Usa frames sintéticos para não depender de câmera ou dataset.

Saída esperada:
```
test_chaves ... ok
test_gray_verdadeira_maior ... ok
test_laplace_verdadeira_maior ... ok
test_treinar_e_classificar ... ok
...
Ran 28 tests in 2.3s  OK
```

---

## Interpretação dos resultados

| Status | Score | Significado |
|--------|-------|-------------|
| **Verdadeira** | ≥ 62% | Cédula provavelmente autêntica |
| **Incerta** | 39–61% | Análise inconclusiva — repetir com melhor iluminação |
| **Falsa** | ≤ 38% | Cédula provavelmente falsa — alerta sonoro emitido |

> ⚠️ Este sistema é uma ferramenta de apoio. A classificação final deve sempre ser confirmada por métodos oficiais do Banco Central do Brasil.

---

## Dicas para melhores resultados

- **Iluminação:** use luz branca uniforme, sem reflexos sobre a cédula.
- **Enquadramento:** centralize a nota e mantenha-a plana e estável.
- **Resolução:** câmeras de 720p ou superior produzem análises mais precisas.
- **Dataset de treino:** recomenda-se ao menos 20 amostras de cada classe para o SVM.
- **Template ORB:** cadastre sempre com a mesma denominação que será verificada (ex.: R$50 com R$50).
- **Modo inicial:** use `COMBINADO` para máxima robustez; use modos individuais para diagnóstico.
