import streamlit as st
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp

st.set_page_config(page_title="Steam Price Helper", page_icon="🎮", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('model/steam_price_helper.pkl')

system = load_model()
model = system['model']
scaler = system['scaler']
dense_features = system['dense_features']
tag_features = system['tag_features']

st.title(" Steam Price Helper 🎮")
st.markdown("Descubra o **preço ideal de lançamento** para o seu jogo utilizando uma regressão treinada com mais de 71 mil títulos do mercado Premium.")
st.divider()

st.sidebar.header("⚙️ Parâmetros do Jogo")
ano = st.sidebar.number_input("Ano de Lançamento", min_value=2024, max_value=2030, value=2026)
mes = st.sidebar.slider("Mês de Lançamento", 1, 12, 6)

exp_dev = st.sidebar.number_input("Experiência da Desenvolvedora", min_value=0, max_value=1000, value=1)
exp_pub = st.sidebar.number_input("Experiência da Publicadora", min_value=0, max_value=1000, value=1)

st.subheader("🏷️ Seleção de Tags (Mecânicas e Nicho)")
tags_selecionadas = st.multiselect(
    "Escolha 5 tags que melhor descrevem o seu jogo:", 
    options=tag_features,
    max_selections=5
)

if st.button("Calcular Preço Sugerido", type="primary"):
    if len(tags_selecionadas) != 5:
        st.error("⚠️ Por favor, selecione exatamente 5 tags antes de calcular o preço.")
    else:
        with st.spinner("Analisando o mercado e calculando..."):
            
            df_densas = pd.DataFrame([[ano, mes, exp_dev, exp_pub]], columns=dense_features)
            densas_escaladas = scaler.transform(df_densas)
            
            vetor_tags = np.zeros((1, len(tag_features)))
            
            lista_impactos = [] 

            for tag in tags_selecionadas:
                idx_tag = tag_features.index(tag)
                vetor_tags[0, idx_tag] = 1.0 
                
                idx_coeficiente = len(dense_features) + idx_tag
                peso_log = model.coef_[idx_coeficiente]
                
                impacto_perc = (np.exp(peso_log) - 1) * 100
                lista_impactos.append({"Tag": tag, "Impacto": impacto_perc})
                    
            tags_esparsas = sp.csr_matrix(vetor_tags)
            
            X_final = sp.hstack((densas_escaladas, tags_esparsas))
            pred_log = model.predict(X_final)[0]
            preco_final_usd = np.expm1(pred_log) / 100.0

            st.success(f"### Preço Sugerido de Lançamento: **${preco_final_usd:.2f} USD**")
            
            st.markdown("#### 📊 Análise de Impacto das Tags Escolhidas:")
            
            lista_impactos = sorted(lista_impactos, key=lambda x: x["Impacto"], reverse=True)
            
            for item in lista_impactos:
                cor_texto = "green" if item["Impacto"] > 0 else "red"
                sinal = "+" if item["Impacto"] > 0 else ""
                st.markdown(f"- **{item['Tag']}**: :{cor_texto}[{sinal}{item['Impacto']:.1f}%]")
                
            st.caption("Nota: Este valor é uma estimativa de teto baseada em atributos mecânicos e cronológicos.")
            st.balloons()
