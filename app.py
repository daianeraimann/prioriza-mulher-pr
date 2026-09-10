from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "base_municipios.xlsx"
GEOJSON_FILE = ROOT / "data" / "municipios_pr.geojson"
IBGE_FILE = ROOT / "data" / "municipios_pr_ibge.json"

st.set_page_config(page_title="Prioriza Mulher PR", page_icon="🟣", layout="wide")


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()
    return text.replace("DOESTE", "DO OESTE")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pct_rank(series: pd.Series, higher_is_priority: bool = True) -> pd.Series:
    s = safe_numeric(series)
    if s.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    rank = s.rank(pct=True, method="average").fillna(0.5)
    return rank if higher_is_priority else 1 - rank


@st.cache_data
def load_base() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "TAXA LESÃO CORPORAL": "Taxa de lesão corporal",
        "TAXA AMEAÇA": "Taxa de ameaça",
        "População  2025": "População 2025",
        "TENTATIVA DE FEMINICÍDIO": "Tentativas de feminicídio",
    }
    df = df.rename(columns=rename)
    df["Município_norm"] = df["Município"].map(normalize_name)
    # A planilha é a fonte principal dos municípios. Nenhuma cidade deve ser
    # excluída apenas porque sua Regional ainda não foi preenchida.
    df = df[~df["Município_norm"].str.contains("TOTAL GERAL", na=False)].copy()
    df = df[df["Município_norm"].ne("")].copy()
    df["Regional de Saúde"] = df["Regional de Saúde"].fillna("Regional não informada")
    df["Macrorregional de Saúde"] = df["Macrorregional de Saúde"].fillna("Macrorregional não informada")
    for col in ["Taxa de lesão corporal", "Taxa de ameaça", "População 2025", "Tentativas de feminicídio", "Quantidade de CAPS"]:
        df[col] = safe_numeric(df[col])
    # A regra de zero vale para municípios ausentes na fonte de tentativas; taxas
    # faltantes permanecem faltantes e recebem posição neutra no score.
    df["Tentativas de feminicídio"] = df["Tentativas de feminicídio"].fillna(0)
    df["Quantidade de CAPS"] = df["Quantidade de CAPS"].fillna(0)
    df["Tentativas por 100 mil hab."] = np.where(
        df["População 2025"] > 0,
        df["Tentativas de feminicídio"] / df["População 2025"] * 100_000,
        0,
    )
    df["Tem Delegacia da Mulher"] = df["Possui Delegacia da Mulher?"].astype(str).str.upper().eq("SIM")
    df["Tem CAPS"] = df["Possui CAPS?"].astype(str).str.upper().eq("SIM")

    ibge = pd.DataFrame(json.loads(IBGE_FILE.read_text(encoding="utf-8")))
    ibge["Município_norm"] = ibge["nome"].map(normalize_name)
    ibge["Código IBGE"] = ibge["id"].astype(str)
    return df.merge(ibge[["Município_norm", "Código IBGE"]], on="Município_norm", how="left")




def score_data(df, weights, strategy):
    out = df.copy()
    out["Violência"] = (
        pct_rank(out["Taxa de lesão corporal"]) + pct_rank(out["Taxa de ameaça"]) + pct_rank(out["Tentativas por 100 mil hab."])
    ) / 3
    out["Vazio de proteção"] = (1 - out["Tem Delegacia da Mulher"].astype(float))
    caps_gap = 1 - pct_rank(out["Quantidade de CAPS"])
    out["Vazio assistencial"] = (caps_gap + (1 - out["Tem CAPS"].astype(float))) / 2
    out["Desconcentração"] = pct_rank(np.log1p(out["População 2025"]), higher_is_priority=False)
    # Evita que municípios muito pequenos liderem só pelo bônus populacional.
    out.loc[out["População 2025"] < 10_000, "Desconcentração"] *= 0.35
    out["Viabilidade"] = np.clip(np.log1p(out["Quantidade de CAPS"]) / np.log(5), 0, 1)

    if strategy == "Vazio assistencial":
        service = (out["Vazio assistencial"] * 0.7 + out["Vazio de proteção"] * 0.3)
    elif strategy == "Implantação rápida":
        service = (out["Viabilidade"] * 0.7 + out["Vazio de proteção"] * 0.3)
    else:
        service = (out["Vazio assistencial"] * 0.35 + out["Viabilidade"] * 0.35 + out["Vazio de proteção"] * 0.30)
    out["Serviços e implantação"] = service

    components = {
        "Violência": weights["Violência"],
        "Serviços e implantação": weights["Serviços"],
        "Desconcentração": weights["Desconcentração"],
    }

    total_weight = max(sum(components.values()), 1)
    out["Score"] = sum(out[c] * w for c, w in components.items()) / total_weight * 100
    out["Ranking estadual"] = out["Score"].rank(ascending=False, method="min").astype(int)
    out["Ranking na Regional"] = out.groupby("Regional de Saúde")["Score"].rank(ascending=False, method="min").astype(int)
    return out


def recommendation_reason(row):
    reasons = []
    if row["Violência"] >= 0.75:
        reasons.append("indicadores de violência elevados")
    if row["Vazio assistencial"] >= 0.65:
        reasons.append("vazio de atenção psicossocial")
    if row["Vazio de proteção"] >= 0.5:
        reasons.append("sem Delegacia da Mulher")
    if row["Viabilidade"] >= 0.5:
        reasons.append("estrutura CAPS que favorece implantação")
    if row["Desconcentração"] >= 0.65:
        reasons.append("favorece desconcentração territorial")
    return "; ".join(reasons[:3]) or "posição relativa equilibrada nos critérios selecionados"


def format_table(df):
    cols = ["Programa sugerido", "Município", "Regional de Saúde", "Macrorregional de Saúde", "Score", "Ranking estadual",
            "Taxa de lesão corporal", "Taxa de ameaça", "Tentativas de feminicídio", "População 2025",
            "Possui CAPS?", "Quantidade de CAPS", "Possui Delegacia da Mulher?", "Justificativa"]
    return df[[c for c in cols if c in df.columns]].sort_values("Score", ascending=False)


def select_balanced_programs(df: pd.DataFrame, number_of_programs: int) -> pd.DataFrame:
    """Seleciona primeiro um município por Regional e completa pelo score."""
    if df.empty or number_of_programs <= 0:
        return df.head(0).copy()

    quantity = min(int(number_of_programs), len(df))
    ordered = df.sort_values("Score", ascending=False)
    regional_winners = (
        ordered.groupby("Regional de Saúde", as_index=False, group_keys=False)
        .head(1)
        .sort_values("Score", ascending=False)
    )
    selected = regional_winners.head(quantity)

    remaining_slots = quantity - len(selected)
    if remaining_slots > 0:
        remaining = ordered.loc[~ordered.index.isin(selected.index)].head(remaining_slots)
        selected = pd.concat([selected, remaining])

    selected = selected.sort_values("Score", ascending=False).copy()
    selected["Programa sugerido"] = range(1, len(selected) + 1)
    return selected


base = load_base()
st.title("Prioriza Mulher PR")
st.caption("Apoio transparente à escolha de municípios para programas de saúde mental destinados a mulheres vítimas de violência")
st.warning("Versão demonstrativa: os dados e resultados ainda estão em fase de conferência e não devem ser utilizados isoladamente para decisões de alocação de recursos.")

with st.expander("Como usar este painel", expanded=True):
    st.markdown("""
1. **Escolha o recorte na barra lateral.** Os filtros apenas retiram ou mantêm municípios na análise; eles não mudam a nota de cada cidade.
2. **Escolha a estratégia.** Ela define se o painel deve valorizar principalmente a falta de serviços, a possibilidade de implantação rápida ou um equilíbrio entre as duas situações.
3. **Ajuste os pesos.** Quanto maior o peso, maior a influência daquele critério no score final. Os pesos são automaticamente transformados em proporções, portanto não precisam somar 100.
4. **Confira a recomendação e o mapa.** A lista pode reservar vagas por Regional de Saúde para evitar concentração somente nos municípios maiores.
5. **Compare municípios.** O gráfico mostra em quais critérios cada cidade se destaca. A decisão final deve considerar também análise técnica e pactuação regional.
""")

with st.expander("O que significa o score?"):
    st.markdown("""
O **score é uma nota de prioridade de 0 a 100**. Quanto maior a nota, maior a prioridade do município segundo as escolhas feitas no painel. Ele engloba três componentes:

- **Violência:** posição relativa do município nas taxas de lesão corporal, ameaça e tentativas de feminicídio por 100 mil habitantes.
- **Serviços e implantação:** considera a existência e a quantidade de CAPS e a existência de Delegacia da Mulher. A forma de combinar esses dados muda conforme a estratégia escolhida.
- **Desconcentração:** favorece a distribuição para além dos maiores centros populacionais. O bônus é reduzido em municípios com menos de 10 mil habitantes para evitar que o pequeno porte, sozinho, gere uma recomendação pouco viável.

Os pesos definem a participação de cada componente. Com os pesos padrão **50, 30 e 20**, violência representa 50% do score, serviços e implantação 30%, e desconcentração 20%. O score compara municípios entre si e **não representa uma probabilidade nem uma quantidade de casos**.
""")

with st.sidebar:
    st.header("Recorte e critérios")
    st.caption("Primeiro defina quais municípios participarão da análise. Deixar um campo vazio significa considerar todas as opções.")
    macros = st.multiselect(
        "Macrorregionais",
        sorted(base["Macrorregional de Saúde"].dropna().unique()),
        help="Ao escolher uma ou mais Macrorregionais, somente os municípios dessas áreas permanecem nos resultados.",
    )
    regional_options = sorted(base.loc[base["Macrorregional de Saúde"].isin(macros), "Regional de Saúde"].unique()) if macros else sorted(base["Regional de Saúde"].dropna().unique())
    regionals = st.multiselect(
        "Regionais de Saúde",
        regional_options,
        help="Restringe a análise às Regionais selecionadas. As opções acompanham o filtro de Macrorregional.",
    )
    pop_range = st.slider(
        "Faixa populacional",
        0,
        int(base["População 2025"].max()),
        (0, int(base["População 2025"].max())),
        step=5_000,
        help="Use este filtro para estudar, por exemplo, somente municípios pequenos ou médios. Municípios fora da faixa deixam de aparecer, mas suas notas não são recalculadas.",
    )
    caps_filter = st.selectbox(
        "Situação quanto ao CAPS",
        ["Todos", "Sem CAPS", "Com CAPS"],
        help="'Sem CAPS' mostra vazios de atendimento. 'Com CAPS' mostra cidades que já possuem estrutura que pode facilitar uma implantação mais rápida.",
    )
    delegacia_filter = st.selectbox(
        "Situação quanto à Delegacia da Mulher",
        ["Todos", "Sem Delegacia", "Com Delegacia"],
        help="Permite analisar somente cidades sem proteção policial especializada ou somente as que já possuem esse serviço.",
    )
    strategy = st.radio(
        "Estratégia de priorização",
        ["Equilibrado", "Vazio assistencial", "Implantação rápida"],
        help="Equilibrado combina carência e estrutura. Vazio assistencial favorece locais com menor oferta. Implantação rápida valoriza a existência de CAPS, que pode apoiar o início do programa.",
    )
    if strategy == "Vazio assistencial":
        st.caption("Resultado esperado: maior prioridade para municípios com pouca estrutura de CAPS e sem Delegacia da Mulher.")
    elif strategy == "Implantação rápida":
        st.caption("Resultado esperado: maior prioridade para municípios que já possuem estrutura CAPS, sem ignorar a ausência de Delegacia da Mulher.")
    else:
        st.caption("Resultado esperado: equilíbrio entre necessidade, estrutura disponível e ausência de serviços especializados.")
    st.subheader("Pesos do score")
    st.caption("O score vai de 0 a 100. Aumentar um peso faz aquele critério influenciar mais a posição dos municípios.")
    w_viol = st.slider("Violência", 0, 100, 50, help="Considera lesão corporal, ameaça e tentativas de feminicídio por 100 mil habitantes. Peso maior favorece cidades com violência relativamente mais alta.")
    w_serv = st.slider("Serviços e implantação", 0, 100, 30, help="Considera CAPS, quantidade de CAPS e Delegacia da Mulher. O efeito exato depende da estratégia escolhida acima.")
    w_desc = st.slider("Desconcentração", 0, 100, 20, help="Peso maior favorece a distribuição para além dos grandes centros. Municípios abaixo de 10 mil habitantes recebem bônus reduzido para evitar indicações pouco viáveis apenas pelo tamanho.")
    weight_total = w_viol + w_serv + w_desc
    if weight_total:
        st.caption(f"Influência atual: violência {w_viol / weight_total:.0%}; serviços {w_serv / weight_total:.0%}; desconcentração {w_desc / weight_total:.0%}.")
    else:
        st.warning("Defina pelo menos um peso acima de zero para produzir uma priorização útil.")
    


scored = score_data(base, {"Violência": w_viol, "Serviços": w_serv, "Desconcentração": w_desc}, strategy)
filtered = scored.copy()
if macros:
    filtered = filtered[filtered["Macrorregional de Saúde"].isin(macros)]
if regionals:
    filtered = filtered[filtered["Regional de Saúde"].isin(regionals)]
filtered = filtered[filtered["População 2025"].isna() | filtered["População 2025"].between(*pop_range)]
if caps_filter != "Todos":
    filtered = filtered[filtered["Tem CAPS"] == (caps_filter == "Com CAPS")]
if delegacia_filter != "Todos":
    filtered = filtered[filtered["Tem Delegacia da Mulher"] == (delegacia_filter == "Com Delegacia")]
filtered["Justificativa"] = filtered.apply(recommendation_reason, axis=1)

tab1, tab2, tab3, tab4 = st.tabs(["Recomendação", "Mapa", "Comparar municípios", "Metodologia e dados"])

with tab1:
    st.markdown("### Municípios sugeridos pelo modelo")
    st.caption("Os indicadores abaixo resumem o recorte criado pelos filtros da barra lateral. Eles não representam o Paraná inteiro quando algum filtro estiver ativo.")
    programs = st.number_input(
        "Quantos programas estão disponíveis para distribuição?",
        min_value=1,
        max_value=max(1, len(filtered)),
        value=min(22, max(1, len(filtered))),
        help="Informe quantos municípios poderão receber o programa. O painel produzirá uma indicação para cada programa, limitada à quantidade de municípios do recorte.",
    )
    shortlist = select_balanced_programs(filtered, int(programs))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Programas disponíveis", int(programs))
    c2.metric("Municípios recomendados", len(shortlist))
    c3.metric("Regionais contempladas", shortlist["Regional de Saúde"].nunique())
    c4.metric("Municípios analisados", len(filtered))

    st.info(
        "Regra de equilíbrio regional: primeiro o painel escolhe o município com maior score de cada Regional de Saúde. "
        "Se ainda houver programas, as vagas restantes vão para os municípios com maiores scores, independentemente da Regional. "
        "Se houver menos programas que Regionais, entram os vencedores regionais com maior score."
    )
    st.caption("A coluna 'Programa sugerido' apenas numera as indicações; não representa ordem de pagamento ou execução.")
    st.markdown(f"#### Proposta de distribuição de {len(shortlist)} programa(s)")
    st.dataframe(format_table(shortlist), use_container_width=True, hide_index=True,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f")})
    csv = format_table(shortlist).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar lista recomendada (CSV)", csv, "municipios_recomendados.csv", "text/csv")

with tab2:
    st.markdown("### Distribuição territorial")
    st.caption("Escolha o dado que será representado pelas cores. Tons mais escuros indicam valores maiores. Passe o mouse sobre um município para ver seus dados.")
    map_explanations = {
        "Score": "Nota geral de prioridade, de 0 a 100, formada por violência, serviços/implantação e desconcentração conforme os pesos escolhidos. Quanto maior, maior a prioridade calculada.",
        "Violência": "Síntese da posição relativa nas taxas de lesão corporal, ameaça e tentativas de feminicídio por 100 mil habitantes. Valor maior indica situação relativamente mais grave na base.",
        "Serviços e implantação": "Combina CAPS, quantidade de CAPS e Delegacia da Mulher. Em 'Vazio assistencial', valor alto indica maior carência; em 'Implantação rápida', maior viabilidade pela estrutura CAPS; em 'Equilibrado', combina as duas perspectivas.",
        "Desconcentração": "Favorece municípios fora dos grandes centros. Valor maior significa maior contribuição para desconcentrar recursos, com redução do bônus abaixo de 10 mil habitantes.",
        "Taxa de lesão corporal": "Taxa de registros de lesão corporal presente na planilha. Tons mais escuros representam taxas maiores.",
        "Taxa de ameaça": "Taxa de registros de ameaça presente na planilha. Tons mais escuros representam taxas maiores.",
        "Tentativas por 100 mil hab.": "Tentativas de feminicídio divididas pela população total e multiplicadas por 100 mil. É uma taxa provisória, pois não utiliza somente a população feminina.",
    }
    map_metric = st.selectbox(
        "O que as cores do mapa devem mostrar?",
        list(map_explanations),
        help="Selecione um indicador. A explicação correspondente aparecerá abaixo.",
    )
    st.info(map_explanations[map_metric])
    geojson = json.loads(GEOJSON_FILE.read_text(encoding="utf-8"))
    map_df = filtered[filtered["Código IBGE"].notna()].copy()
    # OpenStreetMap é gratuito e este estilo não exige token nem chave de API.
    # A malha dos municípios continua sendo lida do arquivo GeoJSON local.
    fig = px.choropleth_mapbox(
        map_df, geojson=geojson, locations="Código IBGE", featureidkey="properties.codarea", color=map_metric,
        hover_name="Município", hover_data={"Código IBGE": False, "Score": ":.1f", "Regional de Saúde": True,
                                             "População 2025": ":,.0f", "Quantidade de CAPS": True},
        color_continuous_scale="RdPu",
        mapbox_style="open-street-map",
        center={"lat": -24.7, "lon": -51.5},
        zoom=5.7,
        opacity=0.72,
        height=650,
    )
    fig.update_traces(marker_line_color="white", marker_line_width=0.35)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_title=map_metric,
        mapbox_accesstoken=None,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Comparação direta")
    st.caption("Adicione de 2 a 8 municípios. Cada linha do gráfico representa uma cidade; quanto mais distante do centro, maior é o valor relativo naquele critério.")
    choices = st.multiselect(
        "Municípios para comparar",
        sorted(filtered["Município"].unique()),
        max_selections=8,
        help="A lista respeita os filtros da barra lateral. Se uma cidade não aparecer, revise a Macrorregional, a Regional, a população, o CAPS ou a Delegacia selecionados.",
    )
    if len(choices) >= 2:
        comp = filtered[filtered["Município"].isin(choices)].copy()
        dimensions = ["Violência", "Serviços e implantação", "Desconcentração", "Vazio de proteção", "Viabilidade"]
        radar = go.Figure()
        for _, row in comp.iterrows():
            vals = [row[d] * 100 for d in dimensions]
            radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=dimensions + [dimensions[0]], fill="toself", name=row["Município"]))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=520, margin=dict(t=30, b=30))
        st.plotly_chart(radar, use_container_width=True)
        st.info("Leitura do gráfico: 'Violência' alta indica maior necessidade relativa; 'Vazio de proteção' alto indica ausência de Delegacia da Mulher; 'Viabilidade' alta indica maior estrutura CAPS; 'Desconcentração' alta favorece municípios fora dos grandes centros.")
        st.dataframe(format_table(comp), use_container_width=True, hide_index=True)
    else:
        st.info("Selecione pelo menos dois municípios para comparar.")

with tab4:
    st.subheader("Como o score funciona")
    st.markdown("""
O score é uma ferramenta de triagem, não uma decisão automática. Cada componente é convertido em posição percentual entre os municípios e combinado pelos pesos da barra lateral. A nota final vai de 0 a 100 e não representa probabilidade.

- **Violência:** média das posições relativas das taxas de lesão corporal, ameaça e tentativas de feminicídio por 100 mil habitantes.
- **Serviços/implantação:** combina CAPS e Delegacia da Mulher. Em **Vazio assistencial**, usa 70% de carência de CAPS e 30% de ausência de Delegacia; em **Implantação rápida**, usa 70% de viabilidade pela estrutura CAPS e 30% de ausência de Delegacia; em **Equilibrado**, combina 35% de carência, 35% de viabilidade e 30% de ausência de Delegacia.
- **Desconcentração:** favorece municípios menores, mas reduz o bônus abaixo de 10 mil habitantes para evitar recomendações inviáveis baseadas apenas no porte.

Na distribuição, o sistema seleciona primeiro o município de maior score de cada Regional de Saúde. As vagas restantes são preenchidas pelos maiores scores ainda não selecionados. Se houver menos programas que Regionais, são escolhidos os vencedores regionais com maior score. Toda alteração de peso, estratégia ou filtro pode mudar o resultado.
""")
    st.warning("Limitações: tentativas de feminicídio foram divididas pela população total, não pela população feminina; os dados representam períodos e fontes diferentes; municípios ausentes na base de tentativas foram tratados como zero; taxas faltantes recebem posição neutra no score. Antes da alocação final, valide período, cobertura, subnotificação, capacidade da rede e pactuação regional.")
    st.subheader("Fontes")
    st.markdown("""
- Base consolidada fornecida pelo projeto (violência, população, Delegacia da Mulher, CAPS e Regionais de Saúde).
- [Regionais de Saúde — SESA/PR](https://www.saude.pr.gov.br/Pagina/Regionais-de-Saude)
- [CAPS — Governo do Paraná](https://www.politicasobredrogas.pr.gov.br/Pagina/Centro-de-Atencao-Psicossocial-CAPS)
- [Delegacias especializadas — TJPR/CEVID](https://www.tjpr.jus.br/web/cevid/delegacias-especializadas)
- [Ipardes — Instituto Paranaense de Desenvolvimento Econômico e Social](https://www.ipardes.pr.gov.br/)
- [Boletim de Ocorrência — Polícia Civil do Paraná](https://www.policiacivil.pr.gov.br/BO)
- Malhas e códigos municipais: IBGE.
""")
    with st.expander("Qualidade e cobertura da base"):
        st.write(f"Municípios carregados diretamente da planilha: {len(base)}.")
        st.caption("A relação de municípios vem da planilha. O arquivo JSON do IBGE é utilizado somente para localizar e desenhar cada município no mapa.")
        missing_codes = base[base["Código IBGE"].isna()]["Município"].tolist()
        if missing_codes:
            st.warning("Municípios da planilha que não puderam ser associados ao mapa: " + ", ".join(missing_codes))
        else:
            st.success("Todos os municípios da planilha foram associados ao mapa.")
