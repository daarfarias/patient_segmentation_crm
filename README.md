# 🏥 Patient Segmentation — Clustering + RFM/RFMT + CLV

Pipeline completo de segmentação de pacientes para redes de clínicas médicas,
combinando clusterização não supervisionada, análise RFM/RFMT e modelagem
probabilística de Customer Lifetime Value (CLV).

## 📌 Visão Geral

Este projeto implementa um pipeline robusto e com checkpoint para segmentar
pacientes com base em seu comportamento de atendimento nos últimos 12 meses.
O objetivo é identificar perfis distintos de pacientes para suportar ações
estratégicas de CRM, retenção e fidelização.

## 🔄 Arquitetura do Pipeline

O pipeline é dividido em 14 etapas, com sistema de checkpoint automático
(via `joblib` + JSON), que permite retomar a execução sem refazer etapas
já concluídas.

| Etapa | Descrição |
|-------|-----------|
| 01 | Leitura e deduplicação da base |
| 02 | Pré-processamento (Imputer mediana + RobustScaler) |
| 03 | Detecção de outliers (IsolationForest, 5% contaminação) |
| 04 | Redução de dimensionalidade (PCA — 90% de variância) |
| 04b | Seleção automática de K (Elbow + Gap Statistic + Consensus Matrix) |
| 05 | Cálculo do eps para DBSCAN (gráfico KNN) |
| 06 | Comparação de algoritmos (KMeans, BIRCH, GMM, DBSCAN) com score ponderado |
| 07 | Treinamento do modelo final + visualização PCA 2D |
| 08 | Análise de pertencimento soft (GMM — confiança por paciente) |
| 09 | Análise RFM com segmentação (Campeões, Fieis, Em Risco, etc.) |
| 10 | Exportação intermediária |
| 11 | Validação de estabilidade (Bootstrap + Jaccard + Hungarian alignment) |
| 12 | Interpretabilidade (Heatmap Z-score + Kruskal-Wallis + SHAP/XGBoost) |
| 13 | RFMT + CLV probabilístico (BG/NBD + Gamma-Gamma via `lifetimes`) |
| 14 | Exportação final completa |

## 🧠 Principais Técnicas

- **Clusterização**: KMeans, BIRCH, Gaussian Mixture Model, DBSCAN
- **Seleção de K**: Elbow Method, Gap Statistic, Consensus Matrix (votação ponderada)
- **Redução dimensional**: PCA adaptativo (componentes para 90% de variância)
- **Detecção de anomalias**: IsolationForest
- **Segmentação RFM**: Recência, Frequência, Monetário (scoring por quintis)
- **Segmentação RFMT**: Extensão com Tenure (tempo de relacionamento)
- **CLV probabilístico**: BG/NBD + Gamma-Gamma (visitas e ticket esperados para 6 meses)
- **Estabilidade**: Bootstrap + índice de Jaccard com alinhamento pelo algoritmo húngaro
- **Interpretabilidade**: Heatmap Z-score, Kruskal-Wallis, Dunn post-hoc, SHAP values

## 📊 Features Utilizadas

22 variáveis comportamentais dos últimos 12 meses, incluindo:
- Dias desde o último atendimento, frequência, valor total e ticket médio
- Quantidade de consultas, exames e procedimentos por categoria
- Participação (share) de cada tipo de serviço
- Taxas de cancelamento e estorno
- Tempo de relacionamento com a clínica

## 📁 Outputs Gerados

**Dados:**
- `resultado_clusters_completo.xlsx/csv` — base completa com cluster, RFM, RFMT e CLV
- `perfil_clusters.xlsx` — perfil médio por cluster
- `resumo_executivo_clusters.xlsx` — resumo gerencial
- `pacientes_anomalos.xlsx` — pacientes identificados como outliers
- `kruskal_wallis_features.xlsx` — features mais discriminantes
- `shap_importancia_features.xlsx` — importância por SHAP
- `dunn_posthoc.xlsx` — testes Dunn por feature significativa

**Visualizações:**
- `clusters_pacientes_pca2d.png` — dispersão PCA 2D dos clusters
- `heatmap_clusters_zscore.png` — perfil Z-score por feature
- `shap_importancia_features.png` — importância global e por cluster
- `estabilidade_bootstrap.png` — histograma e convergência do Jaccard
- `clv_clusters.png` — CLV médio, total, prob. atividade e posicionamento
- `selecao_k_otimo.png` — Elbow, Gap e Consensus para escolha de K
- `pca_variancia.png` e `dbscan_eps.png` — diagnósticos do pipeline

## 🚀 Como Executar

```bash
# 1. Instalar dependências
pip install pandas numpy scikit-learn matplotlib xgboost shap lifetimes scikit-posthocs openpyxl joblib

# 2. Configurar os caminhos da base de dados no início do script
ARQUIVO_DADOS = r'caminho\para\BASE_CLUSTER.xlsx'

# 3. Executar
python clustering_pipeline.py
```

> **Retomada automática**: se a execução for interrompida, basta rodar
> novamente — as etapas já concluídas são reutilizadas via checkpoint.

## 🔧 Dependências
pandas · numpy · scikit-learn · matplotlib · xgboost · shap
lifetimes · scikit-posthocs · scipy · joblib · openpyxl
