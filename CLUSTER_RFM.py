import pandas as pd
import numpy as np
import datetime
import matplotlib
from lifetimes import BetaGeoFitter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal
from scipy import stats
from pathlib import Path
import joblib
import traceback
import json
import xgboost as xgb
import shap
import sys
sys.setrecursionlimit(5000)  # Aumenta de 1000 para 5000 (exemplo)
# Agora rode joblib.dump(resultado, path)

# =========================================================
# SISTEMA DE CHECKPOINT
# =========================================================
CHECKPOINT_DIR = Path("C:\Machine Learning\Clusterizacao\CRM\checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

STATE_FILE = CHECKPOINT_DIR / "pipeline_state.json"


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state_dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state_dict, f, indent=2)


def save_ckpt(name, obj):
    path = CHECKPOINT_DIR / f"{name}.pkl"
    joblib.dump(obj, path)
    print(f"  💾 Checkpoint salvo: {path}")


def load_ckpt(name):
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def etapa_ok(state_dict, nome):
    return state_dict.get(nome) == "ok"


def marcar_ok(state_dict, nome):
    state_dict[nome] = "ok"
    save_state(state_dict)


def run_etapa(state_dict, nome, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"🔄 ETAPA: {nome}")
    resultado_cached = load_ckpt(nome)
    if etapa_ok(state_dict, nome) and resultado_cached is not None:
        print(f"  ✅ Já concluída — reutilizando checkpoint")
        return resultado_cached
    try:
        resultado = fn(*args, **kwargs)
        save_ckpt(nome, resultado)
        marcar_ok(state_dict, nome)
        print(f"  ✅ Etapa concluída com sucesso")
        return resultado
    except Exception as e:
        print(f"  ❌ ERRO na etapa '{nome}':")
        traceback.print_exc()
        print(f"\n  ⚠️  Pipeline interrompido em '{nome}'.")
        print(f"     Corrija o problema e execute novamente.")
        print(f"     As etapas anteriores NÃO serão refeitas.\n")
        raise SystemExit(1)


# =========================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================
ARQUIVO_DADOS = r'C:\Machine Learning\Clusterizacao\CRM\input\BASE_CLUSTER.xlsx'
ARQUIVO_DADOS_CSV = r'C:\Machine Learning\Clusterizacao\CRM\input\BASE_CLUSTER.csv'
output_cluster = r"C:\Machine Learning\Clusterizacao\CRM\output"
output_cluster_etapas = r"C:\Machine Learning\Clusterizacao\CRM\output_etapa"

features_num = [
    "DIAS_ULTIMO_ATENDIMENTO",
    "FREQUENCIA_12M",
    "VALOR_TOTAL_12M",
    "TICKET_MEDIO_12M",
    "TEMPO_RELACIONAMENTO_DIAS",
    "QTD_GRUPOS_12M",
    "QTD_PROCEDIMENTOS_12M",
    "QTD_UNIDADES_12M",
    "QTD_CONSULTAS_12M",
    "QTD_EXAMES_12M",
    "QTD_PROCEDIMENTOS_GRUPO_12M",
    "VALOR_CONSULTAS_12M",
    "VALOR_EXAMES_12M",
    "VALOR_PROCEDIMENTOS_12M",
    "SHARE_CONSULTAS_12M",
    "SHARE_EXAMES_12M",
    "SHARE_PROCEDIMENTOS_12M",
    "QTD_CANCELADAS_12M",
    "QTD_ESTORNADAS_12M",
    "QTD_TOTAL_EVENTOS_12M",
    "TAXA_CANCELAMENTO_12M",
    "TAXA_ESTORNO_12M"
]

id_col = "COD_PACIENTE"
# K será definido automaticamente pela etapa 04b_selecao_k
K = 8  # valor inicial; sobrescrito no __main__ pela etapa 04b


# =========================================================
# ETAPAS 1–10 (ORIGINAIS)
# =========================================================

def _etapa_leitura():
    print("📥 Lendo base...")
    df = pd.read_excel(ARQUIVO_DADOS)
    df = df.drop_duplicates(subset=[id_col]).copy()
    print(f"  Colunas: {list(df.columns)}")
    print(f"  Shape: {df.shape[0]} linhas / {df.shape[1]} colunas")
    return df


def _etapa_preprocessamento(df):
    print("⚙️  Preparando dados...")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, features_num)]
    )
    X = df[features_num]
    X_prepared = preprocessor.fit_transform(X)
    print(f"  Shape preparado: {X_prepared.shape}")
    return {"preprocessor": preprocessor, "X_prepared": X_prepared}


def _etapa_outliers(df, X_prepared):
    print("🔍 Detectando outliers com IsolationForest...")
    iso_forest = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    iso_forest.fit(X_prepared)
    anomaly_scores = iso_forest.decision_function(X_prepared)
    outlier_labels = iso_forest.predict(X_prepared)
    df = df.copy()
    df["anomaly_score"] = anomaly_scores
    df["is_outlier"] = outlier_labels == -1
    df_outliers = df[df["is_outlier"] == True].copy()
    df_clean = df[df["is_outlier"] == False].copy().reset_index(drop=True)
    mask_inliers = outlier_labels == 1
    X_clean = X_prepared[mask_inliers]
    n_out = df_outliers.shape[0]
    print(f"  Anômalos detectados: {n_out} ({n_out / len(df) * 100:.1f}%)")
    return {
        "df_clean": df_clean, "df_outliers": df_outliers,
        "X_clean": X_clean, "iso_forest": iso_forest
    }


def _etapa_pca(X_clean):
    print("📐 Calculando PCA...")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_clean)
    variancia_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(variancia_acumulada) + 1), variancia_acumulada, marker='o')
    plt.axhline(y=0.90, color='red', linestyle='--', label='90% variância')
    plt.xlabel("Número de componentes")
    plt.ylabel("Variância explicada acumulada")
    plt.title("Escolha do número de componentes PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_cluster_etapas}\\pca_variancia.png", dpi=150)
    plt.close()
    N_COMPONENTES = int(np.argmax(variancia_acumulada >= 0.90)) + 1
    print(f"  Componentes para 90% de variância: {N_COMPONENTES}")
    pca = PCA(n_components=N_COMPONENTES, random_state=42)
    X_pca = pca.fit_transform(X_clean)
    print(f"  Shape após PCA: {X_pca.shape}")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {v*100:.1f}%")
    return {"N_COMPONENTES": N_COMPONENTES, "pca": pca, "X_pca": X_pca}


def _etapa_dbscan_eps(X_pca):
    print("📏 Calculando eps para DBSCAN (KNN graph)...")
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_pca)
    distancias, _ = nbrs.kneighbors(X_pca)
    distancias = np.sort(distancias[:, 4])
    plt.figure(figsize=(8, 4))
    plt.plot(distancias)
    plt.axhline(y=distancias[int(len(distancias) * 0.90)],
                color='red', linestyle='--', label='Sugestão de eps (p90)')
    plt.xlabel("Pontos ordenados")
    plt.ylabel("Distância ao 5º vizinho")
    plt.title("Gráfico KNN — Escolha do eps para DBSCAN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_cluster_etapas}\\dbscan_eps.png", dpi=150)
    plt.close()
    eps_sugerido = float(distancias[int(len(distancias) * 0.90)])
    eps_sugerido = max(eps_sugerido, 1e-6)
    print(f"  eps sugerido: {eps_sugerido:.6f}")
    return eps_sugerido


def _etapa_comparacao_algoritmos(X_pca, eps_sugerido):
    print("🚀 Comparando algoritmos de clustering...")
    algoritmos = {
        "KMeans": KMeans(n_clusters=K, random_state=42, n_init=10),
        "BIRCH": Birch(n_clusters=K),
        "GaussianMixture": GaussianMixture(n_components=K, random_state=42, n_init=3),
        "DBSCAN": DBSCAN(eps=eps_sugerido, min_samples=5)
    }
    resultados_alg = []
    modelos_treinados = {}

    def score_final(sil, dbi, ch, n_clusters, n_ruido_pct):
        silhouette_norm = (sil - 0.0) / 1.0
        davies_norm = 1 - (dbi / 2.0)
        calinski_norm = np.log1p(ch) / np.log1p(1e6)
        penalty_ruido = 1 / (1 + n_ruido_pct / 0.1)
        return (0.4 * silhouette_norm + 0.3 * davies_norm + 0.2 * calinski_norm) * penalty_ruido

    for nome, modelo in algoritmos.items():
        try:
            labels = modelo.fit_predict(X_pca)
            modelos_treinados[nome] = modelo
            n_clusters_encontrados = len(set(labels)) - (1 if -1 in labels else 0)
            n_ruido_abs = int(np.sum(labels == -1))
            n_ruido_pct = n_ruido_abs / len(X_pca)
            if n_clusters_encontrados >= 2:
                mask = labels != -1
                sil = silhouette_score(X_pca[mask], labels[mask])
                dbi = davies_bouldin_score(X_pca[mask], labels[mask])
                ch = calinski_harabasz_score(X_pca[mask], labels[mask])
            else:
                sil = dbi = ch = 0
                print(f"  ⚠️  {nome}: não formou clusters válidos")
            score = score_final(sil, dbi, ch, n_clusters_encontrados, n_ruido_pct)
            resultados_alg.append({
                "algoritmo": nome, "n_clusters": n_clusters_encontrados,
                "n_ruido": f"{n_ruido_pct:.1%}", "n_ruido_abs": n_ruido_abs,
                "silhouette ↑": round(sil, 4), "davies_bouldin ↓": round(dbi, 4),
                "calinski_harabasz ↑": f"{ch:,.0f}", "score_final": round(score, 3)
            })
            print(f"  [{nome}] clusters={n_clusters_encontrados} | sil={sil:.4f} | dbi={dbi:.4f} | score={score:.3f}")
        except Exception as e:
            print(f"  ❌ Erro em {nome}: {e}")

    df_resultados = pd.DataFrame(resultados_alg)
    melhor_algoritmo = df_resultados.loc[df_resultados['score_final'].idxmax(), 'algoritmo']
    print(f"\n  🏆 Melhor algoritmo: {melhor_algoritmo} (score: {df_resultados['score_final'].max():.3f})")
    return {
        "df_resultados": df_resultados,
        "modelos_treinados": modelos_treinados,
        "melhor_algoritmo": melhor_algoritmo
    }


def _etapa_modelo_final(X_pca, modelos_treinados, melhor_algoritmo, df_clean, N_COMPONENTES):
    print(f"✅ Aplicando modelo final: {melhor_algoritmo} com K={K}...")
    modelo_final = modelos_treinados[melhor_algoritmo]
    cluster_labels_final = modelo_final.fit_predict(X_pca)
    df_clean = df_clean.copy()
    df_clean["cluster"] = cluster_labels_final
    df_clean["cluster_name"] = df_clean["cluster"].map(
        {i: f"Cluster {i}" for i in range(len(set(cluster_labels_final)))}
    )
    tamanhos = df_clean["cluster"].value_counts().sort_index()
    print(f"\n  📊 Distribuição dos clusters:")
    print(tamanhos)
    minimo = tamanhos.min()
    total = len(df_clean)
    if minimo / total < 0.05:
        print(f"\n  ⚠️  Cluster com apenas {minimo} pacientes ({minimo/total*100:.1f}%) — considere K-1")
    else:
        print(f"\n  ✅ Todos os clusters com tamanho saudável (> 5% do total).")
    if N_COMPONENTES >= 2 and len(X_pca) >= 2:
        pca_2d = PCA(n_components=2, random_state=42)
        X_2d = pca_2d.fit_transform(X_pca)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=cluster_labels_final, cmap='tab10', alpha=0.7, s=20)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variância)")
        plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variância)")
        plt.title(f"Clusters de comportamento de pacientes — {melhor_algoritmo} (K={K})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_cluster_etapas}\\clusters_pacientes_pca2d.png", dpi=150, bbox_inches='tight')
        plt.close()
    return {"df_clean": df_clean, "cluster_labels_final": cluster_labels_final}


def _etapa_gmm(modelos_treinados, X_pca, df_clean):
    print("📊 Análise de pertencimento GMM (soft probabilities)...")
    gmm_modelo = modelos_treinados["GaussianMixture"]
    proba = gmm_modelo.predict_proba(X_pca)
    df_clean = df_clean.copy()
    df_clean["gmm_cluster"] = gmm_modelo.predict(X_pca)
    df_clean["gmm_confianca"] = proba.max(axis=1)
    borderline = df_clean[df_clean["gmm_confianca"] < 0.60]
    print(f"  ⚠️  Pacientes com pertencimento ambíguo (< 60% confiança): {len(borderline)}")
    return df_clean


def _etapa_rfm(df_clean):
    print("🎯 Calculando análise RFM...")

    def calcular_rfm(df, col_recencia, col_frequencia, col_monetario):
        df_rfm = df[[col_recencia, col_frequencia, col_monetario]].copy()
        df_rfm["R_score"] = pd.qcut(df_rfm[col_recencia].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1])
        df_rfm["F_score"] = pd.qcut(df_rfm[col_frequencia].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        df_rfm["M_score"] = pd.qcut(df_rfm[col_monetario].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        df_rfm["R_score"] = df_rfm["R_score"].astype(int)
        df_rfm["F_score"] = df_rfm["F_score"].astype(int)
        df_rfm["M_score"] = df_rfm["M_score"].astype(int)
        df_rfm["RFM_score"] = (df_rfm["R_score"].astype(str) +
                               df_rfm["F_score"].astype(str) + df_rfm["M_score"].astype(str))
        df_rfm["RFM_total"] = df_rfm["R_score"] + df_rfm["F_score"] + df_rfm["M_score"]
        return df_rfm

    def classificar_segmento(row):
        R, F, M = row["R_score"], row["F_score"], row["M_score"]

        if R >= 4 and F >= 4 and M >= 4:
            return "Campeões"
        elif R <= 2 and F >= 4 and M >= 4:
            return "Em risco"  # ← sobe antes de Fieis
        elif F >= 4 and M >= 3:
            return "Fieis"
        elif R <= 2 and F >= 3 and M >= 3:
            return "Ouro perdido"
        elif R >= 4 and F <= 2 and M >= 3:
            return "Potencial Leal"
        elif R >= 4 and F <= 2:
            return "Promissores"
        elif R <= 2 and F <= 2:
            return "Hibernando"
        else:
            return "Regular"

    df_rfm = calcular_rfm(df_clean, "DIAS_ULTIMO_ATENDIMENTO", "FREQUENCIA_12M", "VALOR_TOTAL_12M")
    df_clean = df_clean.copy()
    df_clean["R_score"] = df_rfm["R_score"].values
    df_clean["F_score"] = df_rfm["F_score"].values
    df_clean["M_score"] = df_rfm["M_score"].values
    df_clean["RFM_score"] = df_rfm["RFM_score"].values
    df_clean["RFM_total"] = df_rfm["RFM_total"].values
    df_clean["segmento_rfm"] = df_rfm.apply(classificar_segmento, axis=1)

    print("\n  🔥 Cruzamento Cluster x RFM:")
    print(pd.crosstab(df_clean["cluster"], df_clean["segmento_rfm"], margins=True))
    return df_clean


def _etapa_exportacao(df_clean, df_outliers):
    print("💾 Exportando resultados...")
    perfil = df_clean.groupby("cluster")[features_num].mean().round(2)
    perfil.insert(0, "QTD_PACIENTES", df_clean.groupby("cluster")[features_num[0]].count())

    # ✅ NOVO: coluna de competência
    MES_REFERENCIA = datetime.date.today().replace(day=1).strftime("%d/%m/%Y")
    df_clean = df_clean.copy()
    df_clean["MES"] = MES_REFERENCIA
    df_outliers = df_outliers.copy()
    df_outliers["MES"] = MES_REFERENCIA
    perfil["MES"] = MES_REFERENCIA

    df_clean.to_excel("resultado_clusters_rfm.xlsx", index=False)
    df_outliers.to_excel("pacientes_anomalos.xlsx", index=False)
    perfil.to_excel("perfil_clusters.xlsx")
    print("   → resultado_clusters_rfm.xlsx")
    print("   → pacientes_anomalos.xlsx")
    print("   → perfil_clusters.xlsx")
    return {"perfil": perfil}


# =========================================================
# ETAPA 11 — VALIDAÇÃO DE ESTABILIDADE (BOOTSTRAP + JACCARD)
# =========================================================

def _etapa_estabilidade(X_pca, melhor_algoritmo, n_iter=50, sample_frac=0.8):
    """
    Valida a estabilidade dos clusters via Bootstrap + Jaccard.
    Reamostra os dados N vezes, re-clusteriza e mede a reprodutibilidade
    usando o coeficiente de Jaccard com alinhamento por Hungarian algorithm.

    Interpretação do Jaccard médio:
        >= 0.75 → Clusters estáveis ✅
        0.60–0.75 → Moderadamente estáveis ⚠️
        < 0.60 → Instáveis — revisar K ou algoritmo ❌
    """
    print(f"🔁 Bootstrap stability ({n_iter} iterações, {sample_frac:.0%} dos dados)...")

    def _criar_modelo(seed):
        if melhor_algoritmo == "KMeans":
            return KMeans(n_clusters=K, random_state=seed, n_init=5)
        elif melhor_algoritmo == "BIRCH":
            return Birch(n_clusters=K)
        elif melhor_algoritmo == "GaussianMixture":
            return GaussianMixture(n_components=K, random_state=seed, n_init=2)
        return KMeans(n_clusters=K, random_state=seed, n_init=5)

    # Labels de referência (modelo completo)
    labels_full = _criar_modelo(42).fit_predict(X_pca)
    n = len(X_pca)
    jaccard_list = []

    for i in range(n_iter):
        np.random.seed(i)
        idx = np.random.choice(n, size=int(n * sample_frac), replace=False)
        X_boot = X_pca[idx]
        labels_boot = _criar_modelo(i).fit_predict(X_boot)
        labels_orig_sub = labels_full[idx]

        # Alinhar labels via Hungarian algorithm (contingency matrix)
        k_max = max(labels_orig_sub.max(), labels_boot.max()) + 1
        cont = np.zeros((k_max, k_max))
        for a, b in zip(labels_orig_sub, labels_boot):
            if a >= 0 and b >= 0:
                cont[a, b] += 1

        row_ind, col_ind = linear_sum_assignment(-cont)

        # Jaccard por cluster (TP / (TP + FP + FN))
        jaccards = []
        for r, c in zip(row_ind, col_ind):
            tp = cont[r, c]
            fp = cont[:, c].sum() - tp  # no boot cluster c, not in orig cluster r
            fn = cont[r, :].sum() - tp  # in orig cluster r, not matched to boot cluster c
            denom = tp + fp + fn
            if denom > 0:
                jaccards.append(tp / denom)

        jaccard_list.append(np.mean(jaccards) if jaccards else 0.0)

        if (i + 1) % 10 == 0:
            print(f"  Iter {i+1}/{n_iter} — Jaccard médio acumulado: {np.mean(jaccard_list):.4f}")

    jaccard_mean = float(np.mean(jaccard_list))
    jaccard_std = float(np.std(jaccard_list))

    print(f"\n  📊 Jaccard médio: {jaccard_mean:.4f} ± {jaccard_std:.4f}")
    if jaccard_mean >= 0.75:
        print(f"  ✅ Clusters ESTÁVEIS (Jaccard ≥ 0.75)")
    elif jaccard_mean >= 0.60:
        print(f"  ⚠️  Clusters MODERADAMENTE estáveis (0.60 ≤ Jaccard < 0.75)")
    else:
        print(f"  ❌ Clusters INSTÁVEIS (Jaccard < 0.60) — considere revisar K ou algoritmo")

    # Histograma da distribuição dos Jaccard scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(jaccard_list, bins=20, color='#2E86AB', edgecolor='white', alpha=0.85)
    axes[0].axvline(jaccard_mean, color='red', linestyle='--',
                    linewidth=2, label=f'Média: {jaccard_mean:.3f}')
    axes[0].axvline(0.75, color='green', linestyle=':',
                    linewidth=2, label='Limiar estável (0.75)')
    axes[0].axvline(0.60, color='orange', linestyle=':',
                    linewidth=2, label='Limiar moderado (0.60)')
    axes[0].set_xlabel("Jaccard Score")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title(f"Estabilidade Bootstrap — {melhor_algoritmo} (K={K})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Evolução acumulada da média
    jaccard_cumulative = [np.mean(jaccard_list[:i+1]) for i in range(len(jaccard_list))]
    axes[1].plot(range(1, n_iter + 1), jaccard_cumulative, color='#2E86AB', linewidth=2)
    axes[1].axhline(0.75, color='green', linestyle=':', linewidth=1.5, label='Limiar estável')
    axes[1].axhline(0.60, color='orange', linestyle=':', linewidth=1.5, label='Limiar moderado')
    axes[1].set_xlabel("Iteração Bootstrap")
    axes[1].set_ylabel("Jaccard Médio Acumulado")
    axes[1].set_title("Convergência da Estabilidade")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_cluster_etapas}\\estabilidade_bootstrap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Gráfico salvo: estabilidade_bootstrap.png")

    return {
        "jaccard_scores": jaccard_list,
        "jaccard_mean": jaccard_mean,
        "jaccard_std": jaccard_std
    }


# =========================================================
# ETAPA 12 — INTERPRETABILIDADE: HEATMAP + KRUSKAL + SHAP
# =========================================================

def _etapa_interpretabilidade(df_clean):
    """
    Gera três camadas de interpretabilidade:

    12a — Heatmap Z-score
        Normaliza os perfis médios por feature (Z-score entre clusters).
        Verde = acima da média global, Vermelho = abaixo.
        Facilita comunicar o perfil de cada cluster ao time de CRM.

    12b — Kruskal-Wallis + Dunn post-hoc
        Kruskal-Wallis testa se cada feature discrimina significativamente
        os clusters (H0: todas as medianas são iguais entre clusters).
        Dunn (Bonferroni) identifica quais pares de clusters são diferentes
        entre si para cada feature significativa.

    12c — SHAP values (via XGBoost)
        Treina um classificador XGBoost para prever o label de cluster
        e usa SHAP para medir a contribuição real de cada feature.
        Mais confiável do que importância por permutação.

    Dependências extras: xgboost, shap, scikit-posthocs
        pip install xgboost shap scikit-posthocs
    """
    print("🧠 Gerando interpretabilidade dos clusters...")

    # ── 12a: Heatmap Z-score ──────────────────────────────────────────────
    print("  📊 12a — Heatmap Z-score dos perfis...")
    perfil_media = df_clean.groupby("cluster")[features_num].mean()

    # Z-score por feature (normaliza entre clusters, não entre pacientes)
    perfil_z = perfil_media.apply(stats.zscore, axis=0)

    fig, ax = plt.subplots(figsize=(max(10, K * 1.2), max(8, len(features_num) * 0.42)))
    im = ax.imshow(perfil_z.T, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)
    ax.set_xticks(range(len(perfil_z.index)))
    ax.set_xticklabels([f"Cluster {i}" for i in perfil_z.index], fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(features_num)))
    ax.set_yticklabels(features_num, fontsize=9)

    # Anotações de valor Z nas células
    for i in range(len(perfil_z.index)):
        for j in range(len(features_num)):
            val = perfil_z.iloc[i, j]
            if not np.isnan(val):
                ax.text(i, j, f"{val:.1f}", ha='center', va='center',
                        fontsize=7, color='black' if abs(val) < 1.5 else 'white')

    plt.colorbar(im, ax=ax, label='Z-score (desvios da média global da feature)')
    ax.set_title("Perfil dos Clusters — Z-score por Feature\n"
                 "Verde = acima da média  |  Vermelho = abaixo da média",
                 fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(f"{output_cluster_etapas}\\heatmap_clusters_zscore.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Heatmap Z-score salvo: heatmap_clusters_zscore.png")

    # ── 12b: Kruskal-Wallis + Dunn post-hoc ──────────────────────────────
    print("\n  📐 12b — Kruskal-Wallis por feature...")
    grupos_por_cluster = {
        c: df_clean[df_clean["cluster"] == c][features_num]
        for c in sorted(df_clean["cluster"].unique())
    }
    kruskal_results = []
    for feat in features_num:
        grupos = [g[feat].dropna().values for g in grupos_por_cluster.values()]
        try:
            stat, pval = kruskal(*grupos)
        except Exception:
            stat, pval = np.nan, np.nan
        kruskal_results.append({
            "feature": feat,
            "H_stat": round(stat, 3) if not np.isnan(stat) else np.nan,
            "p_value": pval,
            "p_ajustado_bonferroni": min(pval * len(features_num), 1.0) if not np.isnan(pval) else np.nan
        })

    df_kruskal = pd.DataFrame(kruskal_results).sort_values("p_value")
    df_kruskal["significativa_5pct"] = df_kruskal["p_ajustado_bonferroni"] < 0.05
    df_kruskal.to_excel(f"{output_cluster_etapas}\\kruskal_wallis_features.xlsx", index=False)

    print("\n  🏆 Top 10 features mais discriminantes:")
    print(df_kruskal.head(10)[["feature", "H_stat", "p_value",
                                "p_ajustado_bonferroni", "significativa_5pct"]].to_string(index=False))

    # Dunn post-hoc — top 5 features significativas
    try:
        import scikit_posthocs as sp
        top_features = df_kruskal[df_kruskal["significativa_5pct"]].head(5)["feature"].tolist()
        if top_features:
            print(f"\n  📊 Dunn post-hoc (Bonferroni) nas top {len(top_features)} features:")
            dunn_sheets = {}
            for feat in top_features:
                dunn_df = sp.posthoc_dunn(
                    df_clean, val_col=feat, group_col="cluster", p_adjust="bonferroni"
                )
                dunn_sheets[feat[:31]] = dunn_df  # Excel limita 31 chars por aba
                print(f"\n    Feature: {feat}")
                print(dunn_df.round(4).to_string())

            # Exportar Dunn em Excel multi-abas
            with pd.ExcelWriter(f"{output_cluster_etapas}\\dunn_posthoc.xlsx", engine="openpyxl") as writer:
                for sheet_name, dunn_df in dunn_sheets.items():
                    dunn_df.to_excel(writer, sheet_name=sheet_name)
            print("\n  ✅ Dunn post-hoc exportado: dunn_posthoc.xlsx")
    except ImportError:
        print("  ⚠️  scikit-posthocs não instalado — pulando Dunn post-hoc.")
        print("     Para habilitar: pip install scikit-posthocs")

    # ── 12c: SHAP values ─────────────────────────────────────────────────
    df_shap_importance = pd.DataFrame()
    try:

        print("\n  🔍 12c — Calculando SHAP values com XGBoost...")
        X_feat = df_clean[features_num].fillna(df_clean[features_num].median())
        y = df_clean["cluster"].astype(int)

        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, verbosity=0
        )
        model.fit(X_feat, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_feat)

        shap_por_cluster = {}

        # Multi-class: média do |SHAP| em todas as classes
        # if isinstance(shap_values, list):
        #     mean_shap_global = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        #     # SHAP por cluster
        #     shap_por_cluster = {
        #         i: np.abs(shap_values[i]).mean(axis=0)
        #         for i in range(len(shap_values))
        #     }
        # else:
        #     mean_shap_global = np.abs(shap_values).mean(axis=0)
        #     shap_por_cluster = {}

        # Padroniza shap_values
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)  # (classes, samples, features)

        # Agora trata dimensões
        if shap_values.ndim == 3:
            # Detecta onde estão as features
            if shap_values.shape[0] == len(np.unique(y)):
                # (classes, samples, features)
                shap_values = np.mean(np.abs(shap_values), axis=0)
            else:
                # (samples, features, classes)
                shap_values = np.mean(np.abs(shap_values), axis=2)

        # Agora deve ser (samples, features)
        mean_shap_global = np.mean(np.abs(shap_values), axis=0)

        # Segurança extra (garante 1D)
        mean_shap_global = np.array(mean_shap_global).reshape(-1)

        df_shap_importance = pd.DataFrame({
            "feature": features_num,
            "mean_abs_shap_global": mean_shap_global
        }).sort_values("mean_abs_shap_global", ascending=False)
        df_shap_importance.to_excel(f"{output_cluster_etapas}\\shap_importancia_features.xlsx", index=False)

        # Plot — importância global
        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(features_num) * 0.38)))

        n_top = min(15, len(df_shap_importance))
        top_df = df_shap_importance.head(n_top)
        colors_bar = ['#1565C0' if i < 5 else '#64B5F6' for i in range(n_top)]
        axes[0].barh(top_df["feature"][::-1], top_df["mean_abs_shap_global"][::-1],
                     color=colors_bar[::-1])
        axes[0].set_xlabel("Mean |SHAP value|")
        axes[0].set_title("Importância Global das Features (SHAP)")
        axes[0].grid(True, axis='x', alpha=0.3)

        # SHAP por cluster (heatmap das top features)
        if shap_por_cluster:
            top_feats = df_shap_importance["feature"].head(10).tolist()
            feat_idx = [features_num.index(f) for f in top_feats]
            shap_matrix = np.array([
                shap_por_cluster[c][feat_idx] for c in range(len(shap_por_cluster))
            ])
            im2 = axes[1].imshow(shap_matrix.T, aspect='auto', cmap='Blues')
            axes[1].set_xticks(range(len(shap_por_cluster)))
            axes[1].set_xticklabels([f"C{i}" for i in range(len(shap_por_cluster))], fontsize=9)
            axes[1].set_yticks(range(len(top_feats)))
            axes[1].set_yticklabels(top_feats, fontsize=9)
            plt.colorbar(im2, ax=axes[1], label='Mean |SHAP|')
            axes[1].set_title("SHAP por Cluster (top 10 features)")
        else:
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_cluster_etapas}\\shap_importancia_features.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ SHAP salvo: shap_importancia_features.png + shap_importancia_features.xlsx")

        print("\n  🏆 Top 10 features por SHAP global:")
        print(df_shap_importance.head(10).to_string(index=False))

    except ImportError as e:
        print(f"  ⚠️  SHAP/XGBoost não disponível ({e}).")
        print("     Para habilitar: pip install xgboost shap")

    return {
        "df_kruskal": df_kruskal,
        "df_shap_importance": df_shap_importance,
        "perfil_z": perfil_z
    }


# =========================================================
# ETAPA 13 — RFMT + CLV (BG/NBD + GAMMA-GAMMA)
# =========================================================

def _etapa_rfmt_clv(df_clean):
    """
    Estende o RFM com a dimensão T (Tenure) e calcula CLV via modelos
    probabilísticos BG/NBD + Gamma-Gamma (biblioteca lifetimes).

    RFMT:
        R — Recência (DIAS_ULTIMO_ATENDIMENTO)
        F — Frequência (FREQUENCIA_12M)
        M — Monetário  (VALOR_TOTAL_12M)
        T — Tempo de relacionamento (TEMPO_RELACIONAMENTO_DIAS)

        O T score diferencia um paciente "Promissor Novo" (R alto, F baixo, T baixo)
        de um "Promissor Antigo" (R alto, F baixo, T alto — algo mudou!).
        Isso habilita ações de CRM muito mais precisas.

    CLV (BG/NBD + Gamma-Gamma):
        BG/NBD modela a probabilidade de cada paciente ainda estar ativo
        e estima o número de visitas futuras esperadas.
        Gamma-Gamma modela o valor esperado por visita.
        CLV = transações_esperadas × ticket_esperado

    Dependências: pip install lifetimes openpyxl
    """
    print("💰 Calculando RFMT + CLV (BG/NBD + Gamma-Gamma)...")
    df_clean = df_clean.copy()

    # ── RFMT: T score ─────────────────────────────────────────────────────
    df_clean["T_score"] = pd.qcut(
        df_clean["TEMPO_RELACIONAMENTO_DIAS"].rank(method='first'),
        q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df_clean["RFMT_total"] = (
        df_clean["R_score"] + df_clean["F_score"] +
        df_clean["M_score"] + df_clean["T_score"]
    )

    def classificar_segmento_rfmt(row):
        R, F, M, T = row["R_score"], row["F_score"], row["M_score"], row["T_score"]

        if R >= 4 and F >= 4 and M >= 4:
            return "Campeões"  # melhores em tudo

        elif R >= 3 and F >= 3 and M >= 3 and T >= 3:
            return "Fieis Estabelecidos"  # leais com longo relacionamento

        elif R >= 3 and F >= 3 and M >= 3 and T <= 2:
            return "Fieis em Formação"  # leais, mas relacionamento recente

        elif R >= 4 and F <= 2 and T >= 3:
            return "Promissores Antigos"  # voltou após tempo longo (T=3 incluído)

        elif R >= 4 and F <= 2:
            return "Promissores Novos"  # novo paciente, pouco uso ainda

        elif R <= 2 and F >= 4 and M >= 4:
            return "Em Risco"  # era muito ativo/valioso, mas sumiu

        elif R <= 2 and F >= 3 and M >= 3 and T >= 3:
            return "Ouro Perdido"  # era frequente e valioso, desapareceu

        elif R <= 2 and F <= 2 and T >= 4:
            return "Hibernando Antigo"  # inativo com longo histórico

        elif R <= 2 and F <= 2:
            return "Hibernando"  # inativo sem histórico relevante

        else:
            return "Regular"

    df_clean["segmento_rfmt"] = df_clean.apply(classificar_segmento_rfmt, axis=1)

    print("\n  📊 Distribuição RFMT:")
    print(df_clean["segmento_rfmt"].value_counts())

    print("\n  🔥 Cruzamento Cluster x RFMT:")
    print(pd.crosstab(df_clean["cluster"], df_clean["segmento_rfmt"], margins=True))

    # ── CLV: BG/NBD + Gamma-Gamma ─────────────────────────────────────────
    try:
        print("\n  🔬 Ajustando BG/NBD + Gamma-Gamma...")

        df_ltv = df_clean.copy()
        df_ltv["bgf_T"] = df_ltv["TEMPO_RELACIONAMENTO_DIAS"]
        df_ltv["bgf_frequency"] = (df_ltv["FREQUENCIA_12M"] - 1).clip(lower=0)
        df_ltv["bgf_recency"] = (
                df_ltv["TEMPO_RELACIONAMENTO_DIAS"] - df_ltv["DIAS_ULTIMO_ATENDIMENTO"]
        ).clip(lower=0)
        df_ltv.loc[df_ltv["bgf_frequency"] == 0, "bgf_recency"] = 0
        df_ltv["bgf_recency"] = df_ltv[["bgf_recency", "bgf_T"]].min(axis=1)
        df_ltv["bgf_T"] = df_ltv["TEMPO_RELACIONAMENTO_DIAS"].clip(lower=1)
        df_ltv["bgf_monetary"] = df_ltv["TICKET_MEDIO_12M"].clip(lower=0.01)

        mask_valid = (
                (df_ltv["bgf_T"] > 0) &
                (df_ltv["bgf_recency"] >= 0) &
                (df_ltv["bgf_recency"] <= df_ltv["bgf_T"])
        )
        df_ltv = df_ltv[mask_valid].copy()
        n_invalidos = len(df_clean) - len(df_ltv)
        if n_invalidos > 0:
            print(f"  ⚠️  {n_invalidos} registros removidos por inconsistência temporal (recency > T)")

        # Escala em meses (ajuda convergência)
        df_ltv["bgf_recency"] = df_ltv["bgf_recency"] / 30
        df_ltv["bgf_T"] = df_ltv["bgf_T"] / 30

        # Cap de outliers de frequência
        limite_freq = df_ltv["bgf_frequency"].quantile(0.99)
        df_ltv = df_ltv[df_ltv["bgf_frequency"] <= limite_freq].copy()

        print(f"  📊 Base final CLV: {len(df_ltv):,} registros")

        # ── BG/NBD com retry ──────────────────────────────────────────────
        bgf = None
        for penalizer in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bgf = BetaGeoFitter(penalizer_coef=penalizer)
                    bgf.fit(
                        df_ltv["bgf_frequency"],
                        df_ltv["bgf_recency"],
                        df_ltv["bgf_T"],
                        initial_params=[0.5, 1.0, 0.5, 1.0]
                    )
                print(f"  ✅ BG/NBD convergiu com penalizer={penalizer}")
                break
            except Exception:
                bgf = None

        if bgf is None:
            raise RuntimeError("BG/NBD não convergiu em nenhuma configuração.")

        # Transações esperadas nos próximos 6 meses (escala em meses)
        df_ltv["pred_visitas_6m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            6, df_ltv["bgf_frequency"], df_ltv["bgf_recency"], df_ltv["bgf_T"]
        )

        # Probabilidade de ainda estar ativo
        df_ltv["prob_ativo"] = bgf.conditional_probability_alive(
            df_ltv["bgf_frequency"], df_ltv["bgf_recency"], df_ltv["bgf_T"]
        )

        # ── Gamma-Gamma com retry ─────────────────────────────────────────
        df_gg = df_ltv[df_ltv["bgf_frequency"] > 0].copy()
        ggf = None
        for penalizer in [0.001, 0.01, 0.1, 0.5, 1.0]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ggf = GammaGammaFitter(penalizer_coef=penalizer)
                    ggf.fit(df_gg["bgf_frequency"], df_gg["bgf_monetary"])
                print(f"  ✅ Gamma-Gamma convergiu com penalizer={penalizer}")
                break
            except Exception:
                ggf = None

        if ggf is None:
            print("  ⚠️  Gamma-Gamma não convergiu — usando ticket médio como fallback")
            df_ltv["ticket_esperado"] = df_ltv["bgf_monetary"]
        else:
            df_ltv.loc[df_gg.index, "ticket_esperado"] = ggf.conditional_expected_average_profit(
                df_gg["bgf_frequency"], df_gg["bgf_monetary"]
            )
            df_ltv["ticket_esperado"] = df_ltv["ticket_esperado"].fillna(df_ltv["bgf_monetary"])

        # CLV esperado 6 meses
        df_ltv["clv_6m"] = df_ltv["pred_visitas_6m"] * df_ltv["ticket_esperado"]

        # Merge de volta no df_clean
        cols_merge = [id_col, "pred_visitas_6m", "prob_ativo", "ticket_esperado", "clv_6m"]
        df_clean = df_clean.merge(df_ltv[cols_merge], on=id_col, how="left")

        print(f"\n  ✅ CLV calculado para {len(df_ltv):,} pacientes")

        print("\n  📊 CLV médio por cluster (6 meses):")
        clv_cluster = df_clean.groupby("cluster").agg(
            n_pacientes=("clv_6m", "count"),
            clv_medio=("clv_6m", "mean"),
            clv_mediana=("clv_6m", "median"),
            clv_total=("clv_6m", "sum"),
            prob_ativo_media=("prob_ativo", "mean")
        ).round(2)
        print(clv_cluster)

        # ── Plots CLV ────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = plt.cm.tab10(np.linspace(0, 1, K))

        # 1. CLV médio por cluster
        clv_sorted = clv_cluster["clv_medio"].sort_values(ascending=False)
        axes[0, 0].bar(
            [f"C{i}" for i in clv_sorted.index],
            clv_sorted.values, color=colors[:len(clv_sorted)]
        )
        axes[0, 0].set_title("CLV Médio Esperado — 6 Meses", fontweight='bold')
        axes[0, 0].set_ylabel("CLV Médio (R$)")
        axes[0, 0].grid(True, axis='y', alpha=0.3)

        # 2. Probabilidade de atividade
        prob_sorted = clv_cluster["prob_ativo_media"].sort_values(ascending=False)
        axes[0, 1].bar(
            [f"C{i}" for i in prob_sorted.index],
            prob_sorted.values, color=colors[:len(prob_sorted)]
        )
        axes[0, 1].set_title("Probabilidade de Atividade por Cluster", fontweight='bold')
        axes[0, 1].set_ylabel("Prob. Alive (média)")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, axis='y', alpha=0.3)

        # 3. CLV total por cluster (tamanho do cluster × CLV médio)
        clv_total_sorted = clv_cluster["clv_total"].sort_values(ascending=False)
        axes[1, 0].bar(
            [f"C{i}" for i in clv_total_sorted.index],
            clv_total_sorted.values, color=colors[:len(clv_total_sorted)]
        )
        axes[1, 0].set_title("CLV Total por Cluster (potencial de receita)", fontweight='bold')
        axes[1, 0].set_ylabel("CLV Total (R$)")
        axes[1, 0].grid(True, axis='y', alpha=0.3)

        # 4. Scatter: CLV médio × Prob. Ativo — tamanho = nº pacientes
        scatter_data = clv_cluster.reset_index()
        sizes = scatter_data["n_pacientes"] / scatter_data["n_pacientes"].max() * 1000
        sc = axes[1, 1].scatter(
            scatter_data["prob_ativo_media"],
            scatter_data["clv_medio"],
            s=sizes, alpha=0.8,
            c=scatter_data["cluster"], cmap='tab10'
        )
        for _, row in scatter_data.iterrows():
            axes[1, 1].annotate(
                f"C{int(row['cluster'])}", (row["prob_ativo_media"], row["clv_medio"]),
                fontsize=9, ha='center', va='bottom'
            )
        axes[1, 1].set_xlabel("Probabilidade de Atividade")
        axes[1, 1].set_ylabel("CLV Médio (R$)")
        axes[1, 1].set_title("Posicionamento dos Clusters\n(tamanho = nº pacientes)", fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Customer Lifetime Value por Cluster — BG/NBD + Gamma-Gamma",
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f"{output_cluster_etapas}\\clv_clusters.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Gráficos CLV salvos: clv_clusters.png")

    except ImportError:
        print("  ⚠️  Biblioteca 'lifetimes' não instalada.")
        print("     Para habilitar CLV: pip install lifetimes")
    except Exception as e:
        print(f"  ⚠️  Erro no cálculo CLV: {e}")
        traceback.print_exc()

    return df_clean


# =========================================================
# ETAPA 14 — EXPORTAÇÃO FINAL (ATUALIZADA)
# =========================================================

def _etapa_exportacao_final(df_clean, df_outliers):
    """Exporta todos os resultados, incluindo as novas colunas RFMT e CLV."""
    print("💾 Exportando resultados finais...")
    perfil = df_clean.groupby("cluster")[features_num].mean().round(2)
    perfil.insert(0, "QTD_PACIENTES", df_clean.groupby("cluster")[features_num[0]].count())

    # ✅ NOVO: coluna de competência
    MES_REFERENCIA = datetime.date.today().replace(day=1).strftime("%d/%m/%Y")
    df_clean = df_clean.copy()
    df_clean["MES"] = MES_REFERENCIA
    df_outliers = df_outliers.copy()
    df_outliers["MES"] = MES_REFERENCIA
    perfil["MES"] = MES_REFERENCIA

    df_clean.to_excel(f"{output_cluster}\\resultado_clusters_completo.xlsx", index=False)
    df_outliers.to_excel(f"{output_cluster_etapas}\\pacientes_anomalos.xlsx", index=False)
    perfil.to_excel(f"{output_cluster}\\perfil_clusters.xlsx")
    df_clean.to_csv(f"{output_cluster}\\resultado_clusters_completo.csv", sep=';', index=False, encoding='utf-8-sig')
    # Resumo executivo por cluster
    cols_resumo = ["cluster", "cluster_name", "R_score", "F_score", "M_score",
                   "T_score", "RFMT_total", "segmento_rfmt"]
    if "clv_6m" in df_clean.columns:
        cols_resumo += ["pred_visitas_6m", "prob_ativo", "ticket_esperado", "clv_6m"]
    if "gmm_confianca" in df_clean.columns:
        cols_resumo.append("gmm_confianca")

    resumo = df_clean.groupby("cluster").agg(
        n_pacientes=(id_col, "count"),
        **{c: (c, "mean") for c in cols_resumo
           if c not in ["cluster", "cluster_name", "segmento_rfmt"]
           and c in df_clean.columns}
    ).round(3)
    resumo.to_excel(f"{output_cluster_etapas}\\resumo_executivo_clusters.xlsx", index=False)
    outputs = [
        "resultado_clusters_completo.xlsx",
        "resumo_executivo_clusters.xlsx",
        "pacientes_anomalos.xlsx",
        "perfil_clusters.xlsx",
        "kruskal_wallis_features.xlsx",
        "shap_importancia_features.xlsx",
        "clusters_pacientes_pca2d.png",
        "heatmap_clusters_zscore.png",
        "shap_importancia_features.png",
        "estabilidade_bootstrap.png",
        "clv_clusters.png",
        "pca_variancia.png",
        "dbscan_eps.png",
    ]
    for o in outputs:
        print(f"   → {o}")

    return {"perfil": perfil, "resumo": resumo}


# =========================================================
# EXECUÇÃO PRINCIPAL COM CHECKPOINT
# =========================================================

# =========================================================
# ETAPA 4.5 — SELEÇÃO ÓTIMA DE K (ELBOW + GAP + CONSENSUS)
# =========================================================

def _etapa_selecao_k(X_clean):
    """
    Seleciona o número ótimo de clusters combinando três métodos validados
    pela literatura de clustering.
    Votação ponderada: Gap(40%) + Elbow(30%) + Consensus(30%)
    """
    print("🎯 Seleção automática do número ótimo de clusters...")
    n_max_k = min(15, len(X_clean) // 20)

    # 1. ELBOW METHOD
    inertias = []
    for k in range(2, n_max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(X_clean)
        inertias.append(kmeans.inertia_)
    delta_inertia = np.diff(inertias)
    delta_delta_inertia = np.diff(delta_inertia)
    k_elbow = np.argmax(-delta_delta_inertia) + 2
    k_elbow = max(2, min(k_elbow, n_max_k))

    # 2. GAP STATISTIC
    B = 10
    gaps = np.zeros(n_max_k - 1)
    for b in range(B):
        X_ref = np.random.uniform(np.min(X_clean, axis=0), np.max(X_clean, axis=0), size=X_clean.shape)
        Wk_ref = []
        for k in range(2, n_max_k + 1):
            kmeans_ref = KMeans(n_clusters=k, random_state=b*10+k, n_init=3)
            kmeans_ref.fit(X_ref)
            Wk_ref.append(np.log(kmeans_ref.inertia_))
        gaps += np.array(Wk_ref) - np.log(inertias)
    gaps /= B
    k_gap = 2
    for k in range(2, n_max_k):
        if gaps[k-2] >= gaps[k-1]:
            k_gap = k + 1
        else:
            break

    # 3. CONSENSUS MATRIX (vetorizado, amostra de 2000 pontos)
    sample_size = min(2000, len(X_clean))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_clean), size=sample_size, replace=False)
    X_sample = X_clean[sample_idx]
    n_sample = X_sample.shape[0]  # tamanho real garantido

    n_runs = 20
    consensus = np.zeros((n_sample, n_sample), dtype=np.float32)
    for r in range(n_runs):
        labels = KMeans(n_clusters=8, random_state=r, n_init=3).fit_predict(X_sample)  # ← X_sample, não X_clean
        same = (labels[:, None] == labels[None, :]).astype(np.float32)  # vetorizado
        consensus += same / n_runs
    np.fill_diagonal(consensus, 0)

    consensus_vec = consensus[np.triu_indices(n_sample, k=1)]
    cophenetic_scores = []
    for k in range(2, n_max_k + 1):
        kmeans_cons = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans_cons.fit(consensus_vec.reshape(-1, 1))
        cophenetic_scores.append(kmeans_cons.inertia_)
    delta_coph = np.diff(cophenetic_scores)
    delta_delta_coph = np.diff(delta_coph)
    k_consensus = np.argmax(-delta_delta_coph) + 2

    # VOTAÇÃO FINAL
    k_final = int(0.4 * k_gap + 0.3 * k_elbow + 0.3 * k_consensus)
    k_final = max(2, min(k_final, n_max_k))

    print(f"   Elbow: K={k_elbow}, Gap: K={k_gap}, Consensus: K={k_consensus}")
    print(f"   K FINAL VOTADO: K={k_final} ⭐")

    # Gráfico
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    ax1.plot(range(2, n_max_k + 1), inertias, 'bo-')
    ax1.axvline(k_elbow, color='red', linestyle='--')
    ax1.set_title('Elbow Method')
    ax2.errorbar(range(2, n_max_k + 1), gaps, fmt='go-')
    ax2.axvline(k_gap, color='red', linestyle='--')
    ax2.set_title('Gap Statistic')
    ax3.plot(range(2, n_max_k + 1), cophenetic_scores, 'mo-')
    ax3.axvline(k_consensus, color='red', linestyle='--')
    ax3.set_title('Consensus Matrix')
    ax4.pie([0.4, 0.3, 0.3], labels=['Gap', 'Elbow', 'Consensus'])
    ax4.set_title(f'K Final = {k_final}')
    plt.tight_layout()
    plt.savefig(f'{output_cluster_etapas}\\selecao_k_otimo.png', dpi=150)
    plt.close()

    return {'k_final': k_final}


if __name__ == "__main__":
    print("🚀 Iniciando pipeline de clustering com checkpoint\n")
    print(f"   Checkpoints em: {CHECKPOINT_DIR.resolve()}\n")

    state = load_state()
    etapas_concluidas = [k for k, v in state.items() if v == "ok"]
    if etapas_concluidas:
        print(f"📋 Etapas já concluídas anteriormente: {etapas_concluidas}\n")

    # ── ETAPA 1: Leitura ──────────────────────────────────────────────────
    df = run_etapa(state, "01_leitura", _etapa_leitura)

    # ── ETAPA 2: Preprocessamento ─────────────────────────────────────────
    res_prep = run_etapa(state, "02_preprocessamento", _etapa_preprocessamento, df)
    X_prepared = res_prep["X_prepared"]

    # ── ETAPA 3: Outliers ─────────────────────────────────────────────────
    res_out = run_etapa(state, "03_outliers", _etapa_outliers, df, X_prepared)
    df_clean = res_out["df_clean"]
    df_outliers = res_out["df_outliers"]
    X_clean = res_out["X_clean"]

    # ── ETAPA 4: PCA ──────────────────────────────────────────────────────
    res_pca = run_etapa(state, "04_pca", _etapa_pca, X_clean)
    N_COMPONENTES = res_pca["N_COMPONENTES"]
    X_pca = res_pca["X_pca"]

    # ── ETAPA 4.5: Seleção automática de K ────────────────────────────────
    res_k = run_etapa(state, "04b_selecao_k", _etapa_selecao_k, X_clean)
    K = res_k["k_final"]
    print(f"\n   ✅ K final escolhido automaticamente: {K}")

    # ── ETAPA 5: eps DBSCAN ───────────────────────────────────────────────
    eps_sugerido = run_etapa(state, "05_dbscan_eps", _etapa_dbscan_eps, X_pca)

    # ── ETAPA 6: Comparação de algoritmos ─────────────────────────────────
    res_comp = run_etapa(state, "06_comparacao_algoritmos",
                         _etapa_comparacao_algoritmos, X_pca, eps_sugerido)
    modelos_treinados = res_comp["modelos_treinados"]
    melhor_algoritmo = res_comp["melhor_algoritmo"]

    # ── ETAPA 7: Modelo final + visualização ──────────────────────────────
    res_final = run_etapa(state, "07_modelo_final",
                          _etapa_modelo_final,
                          X_pca, modelos_treinados, melhor_algoritmo,
                          df_clean, N_COMPONENTES)
    df_clean = res_final["df_clean"]

    # ── ETAPA 8: Análise GMM ──────────────────────────────────────────────
    df_clean = run_etapa(state, "08_gmm", _etapa_gmm,
                         modelos_treinados, X_pca, df_clean)

    # ── ETAPA 9: RFM ──────────────────────────────────────────────────────
    df_clean = run_etapa(state, "09_rfm", _etapa_rfm, df_clean)

    # ── ETAPA 10: Exportação intermediária ────────────────────────────────
    run_etapa(state, "10_exportacao_intermediaria", _etapa_exportacao, df_clean, df_outliers)

    # ── ETAPA 11: Estabilidade Bootstrap ─────────────────────────────────
    # NOTA: Esta etapa NÃO é cacheada (resultados são estocásticos por design).
    # Para recalcular com mais iterações, ajuste n_iter abaixo.
    run_etapa(state, "11_estabilidade",
              _etapa_estabilidade, X_pca, melhor_algoritmo,
              n_iter=50, sample_frac=0.8)

    # ── ETAPA 12: Interpretabilidade (Heatmap + Kruskal + SHAP) ──────────
    run_etapa(state, "12_interpretabilidade", _etapa_interpretabilidade, df_clean)

    # ── ETAPA 13: RFMT + CLV (BG/NBD) ────────────────────────────────────
    df_clean = run_etapa(state, "13_rfmt_clv", _etapa_rfmt_clv, df_clean)

    # ── ETAPA 14: Exportação final ────────────────────────────────────────
    run_etapa(state, "14_exportacao_final",
              _etapa_exportacao_final, df_clean, df_outliers)

    print("\n" + "="*60)
    print("🎉 Pipeline concluído com sucesso!")
    print("="*60)
    print("""
📦 Dependências extras necessárias para as novas etapas:
   pip install xgboost shap scikit-posthocs lifetimes openpyxl

💡 Para reiniciar do zero:
   Apague a pasta checkpoints_cluster/

💡 Para reexecutar apenas etapas específicas:
   Apague o .pkl correspondente dentro de checkpoints_cluster/
   (ex: delete 11_estabilidade.pkl para reexecutar com mais iterações)
""")