import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Pricing 2.0 AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SISTEMA DE AUTENTICAÇÃO ---
ARQUIVO_USUARIOS = 'usuarios.json'

def hash_senha(senha):
    """Cria hash SHA-256 da senha"""
    return hashlib.sha256(senha.encode()).hexdigest()

def carregar_usuarios():
    """Carrega usuários do arquivo JSON ou secrets do Streamlit Cloud"""
    # Tenta carregar dos secrets do Streamlit Cloud primeiro
    try:
        if 'USUARIOS_JSON' in st.secrets:
            return json.loads(st.secrets['USUARIOS_JSON'])
    except:
        pass
    
    # Caso contrário, carrega do arquivo local
    if os.path.exists(ARQUIVO_USUARIOS):
        with open(ARQUIVO_USUARIOS, 'r') as f:
            return json.load(f)
    return {}

def verificar_credenciais(usuario, senha):
    """Verifica se as credenciais são válidas"""
    usuarios = carregar_usuarios()
    if usuario in usuarios:
        senha_hash = hash_senha(senha)
        if usuarios[usuario]['senha_hash'] == senha_hash:
            return True, usuarios[usuario]
    return False, None

def mostrar_login():
    """Exibe tela de login"""
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        st.title("🔐 Pricing 2.0 AI")
        st.markdown("### Login")
        
        with st.form("login_form"):
            usuario = st.text_input("Usuário", placeholder="Digite seu usuário")
            senha = st.text_input("Senha", type="password", placeholder="Digite sua senha")
            submit = st.form_submit_button("Entrar", use_container_width=True)
            
            if submit:
                if usuario and senha:
                    valido, dados_usuario = verificar_credenciais(usuario, senha)
                    if valido:
                        st.session_state['autenticado'] = True
                        st.session_state['usuario'] = usuario
                        st.session_state['nome_usuario'] = dados_usuario['nome']
                        st.session_state['nivel'] = dados_usuario['nivel']
                        st.success("✅ Login realizado com sucesso!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Usuário ou senha incorretos!")
                else:
                    st.warning("⚠️ Preencha todos os campos!")
        
        st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)

def fazer_logout():
    """Realiza logout do usuário"""
    st.session_state['autenticado'] = False
    st.session_state['usuario'] = None
    st.session_state['nome_usuario'] = None
    st.session_state['nivel'] = None
    st.rerun()

# Inicializa session state
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

# Verifica autenticação
if not st.session_state['autenticado']:
    mostrar_login()
    st.stop()

# Se autenticado, mostra informações do usuário na sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"👤 **{st.session_state['nome_usuario']}**")
    st.caption(f"Nível: {st.session_state['nivel']}")
    if st.button("🚪 Sair", use_container_width=True):
        fazer_logout()
    st.markdown("---")

# --- VARIÁVEIS GLOBAIS ---
DB_NAME = 'vendas_historico.db'
ARQUIVO_MESTRE = 'dados_limpos_para_ia.csv'

FEATURES = [
    'custo_unitario', 'markup', 'ratio_competitividade', 'mes_num',
    'taxa_conversao', 'media_historica', 'performance_categoria', 'custo_ads'
]
TARGET = 'qtd_venda_7d'
MONOTONIC_CONSTRAINTS = [0, -1, -1, 0, 1, 1, 1, 1]

# ==============================================================================
# 1. MÓDULO DE DADOS (ETL + SQLITE)
# ==============================================================================
def limpar_dataframe(df):
    """Realiza a limpeza pesada e padronização dos dados brutos."""
    
    # Renomear
    mapa = {
        'Codigo': 'id_produto', 'B1 Desc': 'nome_produto', 'descrição': 'nome_produto', 
        'Day of DATA': 'data_referencia', 'Mês de Day of DATA': 'mes',
        'DIF PRECO': 'dif_preco_concorrente', 'custo': 'custo_unitario',
        'preço atacado': 'preco_atacado', 'preço atacado concorrente': 'preco_atacado_concorrente',
        'folga': 'margem_valor', 'un.7d varejo': 'qtd_venda_7d',
        '7dd 002 COM PAI': 'qtd_venda_ads', 'preço varejo': 'preco_varejo',
        'varejo concorrente': 'preco_varejo_concorrente', 'pgv total': 'faturamento_bruto',
        'Custo ADS com PAI': 'custo_ads', 'tx conv aprox': 'taxa_conversao',
        'media 30 dias': 'target_media_30d', 'estoque': 'estoque_atual',
        'Categoria': 'categoria', 'categoria': 'categoria', 'Departamento': 'categoria'
    }
    df = df.rename(columns=mapa)
    
    # Categoria
    if 'categoria' not in df.columns: df['categoria'] = 'GERAL'
    else: df['categoria'] = df['categoria'].fillna('OUTROS').astype(str).str.upper().str.strip()

    # Datas (Blindagem Anti-1970)
    if 'data_referencia' in df.columns:
        if pd.api.types.is_numeric_dtype(df['data_referencia']):
            df['data_referencia'] = pd.to_datetime(df['data_referencia'], unit='D', origin='1899-12-30')
        else:
            df['data_referencia'] = pd.to_datetime(df['data_referencia'], errors='coerce', dayfirst=True)
        
        df = df.dropna(subset=['data_referencia'])
        df = df[df['data_referencia'].dt.year >= 2020]
        df = df.sort_values('data_referencia')

    # Numéricos
    cols_num = ['custo_unitario', 'preco_varejo', 'preco_varejo_concorrente', 
                'qtd_venda_7d', 'custo_ads', 'estoque_atual', 'target_media_30d']
    
    for col in cols_num:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('R$', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Conversão
    if 'taxa_conversao' in df.columns:
        if df['taxa_conversao'].dtype == 'object':
            df['taxa_conversao'] = df['taxa_conversao'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.')
        df['taxa_conversao'] = pd.to_numeric(df['taxa_conversao'], errors='coerce').fillna(0)

    # Engenharia Básica para o Banco
    if 'preco_varejo_concorrente' in df.columns:
        df['preco_varejo_concorrente'] = df['preco_varejo_concorrente'].replace(0, np.nan).fillna(df['preco_varejo'])
    
    if 'ratio_competitividade' not in df.columns and 'preco_varejo' in df.columns:
        df['ratio_competitividade'] = df['preco_varejo'] / df['preco_varejo_concorrente']

    if 'mes_num' not in df.columns and 'data_referencia' in df.columns:
        df['mes_num'] = df['data_referencia'].dt.month

    return df

def processar_upload(uploaded_file):
    """Recebe o arquivo do Streamlit, atualiza o SQLite e gera o CSV da IA."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df_novo = pd.read_csv(uploaded_file)
        else:
            df_novo = pd.read_excel(uploaded_file)
        
        df_limpo = limpar_dataframe(df_novo)
        
        conn = sqlite3.connect(DB_NAME)
        try:
            df_historico = pd.read_sql("SELECT * FROM vendas", conn)
            df_historico['data_referencia'] = pd.to_datetime(df_historico['data_referencia'])
        except:
            df_historico = pd.DataFrame()
            
        df_final = pd.concat([df_historico, df_limpo])
        
        if 'id_produto' in df_final.columns and 'data_referencia' in df_final.columns:
            df_final['id_str'] = df_final['id_produto'].astype(str)
            df_final = df_final.drop_duplicates(subset=['id_str', 'data_referencia'], keep='last')
            df_final = df_final.drop(columns=['id_str'])
            
        df_final.to_sql('vendas', conn, if_exists='replace', index=False)
        df_final.to_csv(ARQUIVO_MESTRE, index=False)
        conn.close()
        
        return True, len(df_limpo), len(df_final)
    
    except Exception as e:
        return False, str(e), 0

# ==============================================================================
# 2. MÓDULO DE IA (Treinamento com Cache)
# ==============================================================================
@st.cache_resource(show_spinner="Treinando Cérebro da IA...")
def treinar_modelo():
    if not os.path.exists(ARQUIVO_MESTRE):
        return None, None, None, None

    try:
        df = pd.read_csv(ARQUIVO_MESTRE)
        if len(df) == 0: return None, None, None, None
    except:
        return None, None, None, None
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '')
    
    # Recalcula colunas essenciais
    if 'custo_unitario' in df.columns and 'preco_varejo' in df.columns:
        custo_safe = df['custo_unitario'].replace(0, 0.01)
        df['markup'] = df['preco_varejo'] / custo_safe
    else:
        df['markup'] = 0

    if 'preco_varejo_concorrente' not in df.columns:
        df['preco_varejo_concorrente'] = df['preco_varejo']
    else:
        df['preco_varejo_concorrente'] = df['preco_varejo_concorrente'].fillna(df['preco_varejo'])
    
    if 'ratio_competitividade' not in df.columns:
        concorrente_safe = df['preco_varejo_concorrente'].replace(0, np.nan).fillna(df['preco_varejo'])
        df['ratio_competitividade'] = df['preco_varejo'] / concorrente_safe

    # CORREÇÃO: Não preencher media_historica e performance_categoria com 0 aqui
    cols_zero = ['taxa_conversao', 'custo_ads'] 
    for c in cols_zero:
        if c not in df.columns: df[c] = 0
        else: df[c] = df[c].fillna(0)

    df_t = df[
        (df['estoque_atual'] > 0) & 
        (df['markup'] > 0) & 
        (df['markup'] < 10) 
    ].copy()
    
    if TARGET not in df_t.columns: return None, None, None, None
    
    df_t[TARGET] = pd.to_numeric(df_t[TARGET], errors='coerce').fillna(0)
    df_t = df_t.dropna(subset=[TARGET])

    if len(df_t) == 0: return None, None, None, None

    # Feature Engineering (Memória)
    if 'media_historica' in df_t.columns: df_t.drop(columns=['media_historica'], inplace=True)
    if 'performance_categoria' in df_t.columns: df_t.drop(columns=['performance_categoria'], inplace=True)
    if 'media_historica' in df.columns: df.drop(columns=['media_historica'], inplace=True)
    if 'performance_categoria' in df.columns: df.drop(columns=['performance_categoria'], inplace=True)

    df_base = df_t.groupby('id_produto')[TARGET].mean().reset_index().rename(columns={TARGET: 'media_historica'})
    df_cat = df_t.groupby('categoria')[TARGET].mean().reset_index().rename(columns={TARGET: 'performance_categoria'})
    
    df_t = df_t.merge(df_base, on='id_produto', how='left').merge(df_cat, on='categoria', how='left')
    df = df.merge(df_base, on='id_produto', how='left').merge(df_cat, on='categoria', how='left')
    
    # Preenche NaNs gerados pelo merge (novos produtos sem histórico ficam com 0)
    df_t['media_historica'] = df_t['media_historica'].fillna(0)
    df_t['performance_categoria'] = df_t['performance_categoria'].fillna(0)
    df['media_historica'] = df['media_historica'].fillna(0)
    df['performance_categoria'] = df['performance_categoria'].fillna(0)
    
    # ABC Class
    df_t['fat'] = df_t[TARGET] * df_t['preco_varejo']
    resumo = df_t.groupby('id_produto')['fat'].sum().sort_values(ascending=False)
    acumulado = resumo.cumsum() / resumo.sum()
    abc_map = acumulado.apply(lambda x: 'A' if x<=0.8 else ('B' if x<=0.95 else 'C')).to_dict()

    df_t['target_log'] = np.log1p(df_t[TARGET])
    
    # Agora as colunas existem e são únicas, o dropna funciona
    df_t = df_t.dropna(subset=FEATURES + ['target_log'])

    if len(df_t) == 0: return None, None, None, None

    modelo = HistGradientBoostingRegressor(
        max_iter=100, monotonic_cst=MONOTONIC_CONSTRAINTS, random_state=42
    )
    modelo.fit(df_t[FEATURES], df_t['target_log'])
    
    return modelo, df, df_t, abc_map

# ==============================================================================
# 3. INTERFACE (STREAMLIT)
# ==============================================================================

st.sidebar.title("💎 Pricing 2.0")
menu = st.sidebar.radio("Navegação", ["Otimização (Lista)", "Auditoria IA", "Banco de Dados"])

# Tenta carregar, mas não trava se falhar
modelo, df_main, df_train, abc_map = treinar_modelo()

# --- TRAVA DE SEGURANÇA SELETIVA ---
if modelo is None:
    if menu == "Banco de Dados":
        pass 
    else:
        st.warning("⚠️ Nenhum dado encontrado. A IA precisa de dados históricos para funcionar.")
        st.info("👈 Clique na aba 'Banco de Dados' no menu à esquerda e faça o upload da sua primeira planilha.")
        st.stop() 



# --- ABA: OTIMIZAÇÃO ---
elif menu == "Otimização (Lista)":
    st.title("🚀 Sugestões de Preço Inteligentes")
    st.info("A IA está analisando cada produto para encontrar o preço ótimo que maximiza o lucro.")
    
    if st.button("Gerar Sugestões Agora"):
        with st.spinner("Otimizando linha a linha... (Isso pode levar alguns segundos)"):
            df_opt = df_main.sort_values('data_referencia').groupby('id_produto').tail(1).copy()
            df_opt['Classe ABC'] = df_opt['id_produto'].map(abc_map).fillna('C')
            
            sugestoes = []
            
            for index, row in df_opt.iterrows():
                custo_prod = row['custo_unitario']
                preco_atual = row['preco_varejo']
                concorrente = row.get('preco_varejo_concorrente', 0)
                conv = row.get('taxa_conversao', 0)
                ads_spend = row.get('custo_ads', 0)
                mes = row['mes_num']
                hist_medio = row.get('media_historica', 0)
                cat_perf = row.get('performance_categoria', 0)
                
                venda_atual_real = row[TARGET] if TARGET in row else 0
                
                if pd.isna(custo_prod) or custo_prod <= 0 or preco_atual <= 0: continue
                
                try:
                    min_price = int(max(custo_prod * 1.05, preco_atual * 0.85))
                    max_price = int(preco_atual * 1.25)
                except: continue
                
                if max_price <= min_price: continue
                
                # Cria range de preços para testar
                precos_teste = np.arange(min_price, max_price + 1, 1)
                
                # Inclui o preço atual para comparação
                if int(preco_atual) not in precos_teste:
                    precos_teste = np.append(precos_teste, int(preco_atual))
                precos_teste = np.sort(precos_teste)
                
                # Monta batch para previsão
                batch_teste = pd.DataFrame({
                    'custo_unitario': custo_prod,
                    'markup': precos_teste / custo_prod,
                    'ratio_competitividade': precos_teste / (concorrente if concorrente > 0 else preco_atual),
                    'mes_num': mes,
                    'taxa_conversao': conv,
                    'media_historica': hist_medio,
                    'performance_categoria': cat_perf,
                    'custo_ads': ads_spend 
                })
                
                vendas_log = modelo.predict(batch_teste[FEATURES])
                vendas_previstas = np.expm1(vendas_log)
                vendas_previstas = np.maximum(vendas_previstas, 0)
                
                lucros_previstos = ((precos_teste - custo_prod) * vendas_previstas) - ads_spend
                
                # Pega métricas do preço atual
                idx_atual = np.where(precos_teste == int(preco_atual))[0]
                lucro_atual_estimado = lucros_previstos[idx_atual[0]] if len(idx_atual) > 0 else 0
                
                idx_melhor = np.argmax(lucros_previstos)
                
                if lucros_previstos[idx_melhor] > 0:
                    sugerido = precos_teste[idx_melhor]
                    
                    status = "Normal"
                    confianca = "Alta"
                    
                    # Lógica de Confiança e Status
                    historico_prod = df_train[df_train['id_produto'] == row['id_produto']]
                    max_price_seen = historico_prod['preco_varejo'].max() if len(historico_prod) > 0 else preco_atual * 1.5
                    
                    venda_max_p = vendas_previstas[-1]
                    venda_min_p = vendas_previstas[0]
                    sensibilidade = ((venda_min_p - venda_max_p) / venda_min_p) * 100 if venda_min_p > 0.01 else 0
                    
                    if sugerido > max_price_seen * 1.10:
                        confianca = "Baixa"
                        status = "⚠️ Preço Inédito"
                    elif sensibilidade < 0.5: 
                        status = "⚠️ Demanda Rígida"
                    elif ((sugerido - preco_atual) / preco_atual) < 0:
                        status = "📉 Ganhar Volume"
                    else:
                        status = "📈 Ganhar Margem"

                    sugestoes.append({
                        'ID': str(row['id_produto']),
                        'Produto': str(row['nome_produto']),
                        'Classe ABC': row['Classe ABC'],
                        'Preço Atual': preco_atual,
                        'Preço Sugerido': sugerido,
                        'Vendas Atuais': venda_atual_real, # COLUNA NOVA
                        'Vendas Previstas': vendas_previstas[idx_melhor],
                        'Lucro Atual': lucro_atual_estimado, # COLUNA NOVA
                        'Lucro Otimizado': lucros_previstos[idx_melhor],
                        'Uplift Financeiro': lucros_previstos[idx_melhor] - lucro_atual_estimado,
                        'Confiança': confianca,
                        'Status': status
                    })
            
            df_resultado = pd.DataFrame(sugestoes)
            
            # Adiciona categoria ao df_resultado
            df_resultado = df_resultado.merge(
                df_opt[['id_produto', 'categoria']].drop_duplicates(),
                left_on='ID',
                right_on='id_produto',
                how='left'
            ).drop(columns=['id_produto'], errors='ignore')
            
            # Preenche categoria faltante
            if 'categoria' not in df_resultado.columns:
                df_resultado['categoria'] = 'GERAL'
            else:
                df_resultado['categoria'] = df_resultado['categoria'].fillna('GERAL')
            
            # ========== PAINEL DE ANÁLISE ==========
            st.success(f"✅ Otimização concluída! {len(df_resultado)} sugestões geradas.")
            
            st.markdown("---")
            st.header("📊 Análise de Impacto da Otimização")
            
            # --- KPIs TOTAIS ---
            st.markdown("### 💰 Resumo Geral")
            
            total_lucro_atual = df_resultado['Lucro Atual'].sum()
            total_lucro_otimizado = df_resultado['Lucro Otimizado'].sum()
            ganho_total = total_lucro_otimizado - total_lucro_atual
            percentual_ganho = (ganho_total / total_lucro_atual * 100) if total_lucro_atual > 0 else 0
            
            total_vendas_atuais = df_resultado['Vendas Atuais'].sum()
            total_vendas_previstas = df_resultado['Vendas Previstas'].sum()
            diff_vendas = total_vendas_previstas - total_vendas_atuais
            
            preco_medio_atual = (df_resultado['Preço Atual'] * df_resultado['Vendas Atuais']).sum() / total_vendas_atuais if total_vendas_atuais > 0 else 0
            preco_medio_otim = (df_resultado['Preço Sugerido'] * df_resultado['Vendas Previstas']).sum() / total_vendas_previstas if total_vendas_previstas > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Lucro Atual",
                    f"R$ {total_lucro_atual:,.2f}",
                    help="Lucro total com preços atuais"
                )
            
            with col2:
                st.metric(
                    "Lucro Otimizado",
                    f"R$ {total_lucro_otimizado:,.2f}",
                    delta=f"+{percentual_ganho:.1f}%",
                    help="Lucro projetado com preços sugeridos"
                )
            
            with col3:
                st.metric(
                    "Ganho Potencial",
                    f"R$ {ganho_total:,.2f}",
                    delta=f"R$ {ganho_total:,.2f}",
                    help="Diferença entre lucro otimizado e atual"
                )
            
            with col4:
                st.metric(
                    "Unidades (7d)",
                    f"{int(total_vendas_previstas):,}",
                    delta=f"{int(diff_vendas):+,}",
                    help="Total de unidades previstas vs atuais"
                )
            
            # Métricas Secundárias
            col5, col6 = st.columns(2)
            
            with col5:
                st.metric(
                    "Preço Médio Atual",
                    f"R$ {preco_medio_atual:,.2f}"
                )
            
            with col6:
                st.metric(
                    "Preço Médio Sugerido",
                    f"R$ {preco_medio_otim:,.2f}",
                    delta=f"{((preco_medio_otim - preco_medio_atual) / preco_medio_atual * 100) if preco_medio_atual > 0 else 0:.1f}%"
                )
            
            st.markdown("---")
            
            # --- ANÁLISE POR CATEGORIA ---
            st.markdown("### 📦 Análise por Categoria")
            
            resumo_cat = df_resultado.groupby('categoria').agg({
                'Lucro Atual': 'sum',
                'Lucro Otimizado': 'sum',
                'Uplift Financeiro': 'sum',
                'Vendas Atuais': 'sum',
                'Vendas Previstas': 'sum',
                'ID': 'count'
            }).reset_index()
            
            resumo_cat.columns = ['Categoria', 'Lucro Atual', 'Lucro Otimizado', 'Ganho', 'Vendas Atuais', 'Vendas Otimizadas', 'Produtos']
            resumo_cat['% Ganho'] = (resumo_cat['Ganho'] / resumo_cat['Lucro Atual'] * 100).fillna(0)
            resumo_cat = resumo_cat.sort_values('Ganho', ascending=False)
            
            for idx, cat_row in resumo_cat.iterrows():
                cat_nome = cat_row['Categoria']
                cat_ganho = cat_row['Ganho']
                cat_perc = cat_row['% Ganho']
                
                with st.expander(f"🏷️ {cat_nome} - Ganho: R$ {cat_ganho:,.2f} ({cat_perc:+.1f}%)"):
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.metric("Lucro Atual", f"R$ {cat_row['Lucro Atual']:,.2f}")
                        st.metric("Vendas Atuais", f"{int(cat_row['Vendas Atuais']):,} un")
                    
                    with c2:
                        st.metric("Lucro Otimizado", f"R$ {cat_row['Lucro Otimizado']:,.2f}")
                        st.metric("Vendas Otimizadas", f"{int(cat_row['Vendas Otimizadas']):,} un")
                    
                    with c3:
                        st.metric("Ganho Potencial", f"R$ {cat_ganho:,.2f}", delta=f"+{cat_perc:.1f}%")
                        st.metric("Total de Produtos", f"{int(cat_row['Produtos'])}")
            
            st.markdown("---")
            
            # --- ANÁLISE POR CURVA ABC ---
            st.markdown("### 🎯 Análise por Curva ABC")
            
            resumo_abc = df_resultado.groupby('Classe ABC').agg({
                'Lucro Atual': 'sum',
                'Lucro Otimizado': 'sum',
                'Uplift Financeiro': 'sum',
                'Vendas Atuais': 'sum',
                'Vendas Previstas': 'sum',
                'ID': 'count'
            }).reset_index()
            
            resumo_abc.columns = ['Classe ABC', 'Lucro Atual', 'Lucro Otimizado', 'Ganho', 'Vendas Atuais', 'Vendas Otimizadas', 'Produtos']
            resumo_abc['% Ganho'] = (resumo_abc['Ganho'] / resumo_abc['Lucro Atual'] * 100).fillna(0)
            
            # Ordena A, B, C
            ordem_abc = {'A': 0, 'B': 1, 'C': 2}
            resumo_abc['ordem'] = resumo_abc['Classe ABC'].map(ordem_abc)
            resumo_abc = resumo_abc.sort_values('ordem').drop(columns=['ordem'])
            
            for idx, abc_row in resumo_abc.iterrows():
                abc_classe = abc_row['Classe ABC']
                abc_ganho = abc_row['Ganho']
                abc_perc = abc_row['% Ganho']
                
                emoji_abc = {'A': '🥇', 'B': '🥈', 'C': '🥉'}
                color_abc = {'A': '🟡', 'B': '⚪', 'C': '🟤'}
                
                with st.expander(f"{emoji_abc.get(abc_classe, '')} Classe {abc_classe} - Ganho: R$ {abc_ganho:,.2f} ({abc_perc:+.1f}%)"):
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.metric("Lucro Atual", f"R$ {abc_row['Lucro Atual']:,.2f}")
                        st.metric("Vendas Atuais", f"{int(abc_row['Vendas Atuais']):,} un")
                    
                    with c2:
                        st.metric("Lucro Otimizado", f"R$ {abc_row['Lucro Otimizado']:,.2f}")
                        st.metric("Vendas Otimizadas", f"{int(abc_row['Vendas Otimizadas']):,} un")
                    
                    with c3:
                        st.metric("Ganho Potencial", f"R$ {abc_ganho:,.2f}", delta=f"+{abc_perc:.1f}%")
                        st.metric("Total de Produtos", f"{int(abc_row['Produtos'])}")
            
            st.markdown("---")
            
            # --- TOP PRODUTOS COM MAIOR GANHO ---
            st.markdown("### 🏆 Top 10 Produtos com Maior Ganho Potencial")
            
            top10_ganho = df_resultado.nlargest(10, 'Uplift Financeiro')[
                ['Produto', 'Classe ABC', 'categoria', 'Preço Atual', 'Preço Sugerido', 
                 'Lucro Atual', 'Lucro Otimizado', 'Uplift Financeiro']
            ].copy()
            
            top10_ganho['% Ganho'] = (top10_ganho['Uplift Financeiro'] / top10_ganho['Lucro Atual'] * 100).fillna(0)
            
            st.dataframe(
                top10_ganho,
                column_config={
                    "Produto": "Produto",
                    "Classe ABC": "ABC",
                    "categoria": "Categoria",
                    "Preço Atual": st.column_config.NumberColumn("Preço Atual", format="R$ %.2f"),
                    "Preço Sugerido": st.column_config.NumberColumn("Preço Sugerido", format="R$ %.2f"),
                    "Lucro Atual": st.column_config.NumberColumn("Lucro Atual", format="R$ %.2f"),
                    "Lucro Otimizado": st.column_config.NumberColumn("Lucro Otimizado", format="R$ %.2f"),
                    "Uplift Financeiro": st.column_config.NumberColumn("Ganho", format="R$ %.2f"),
                    "% Ganho": st.column_config.NumberColumn("% Ganho", format="%.1f%%")
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.markdown("---")
            
            # --- TABELA COMPLETA DE SUGESTÕES ---
            st.markdown("### 📋 Tabela Completa de Sugestões")
            
            st.dataframe(
                df_resultado,
                column_config={
                    "Preço Atual": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Preço Sugerido": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Lucro Atual": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Lucro Otimizado": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Uplift Financeiro": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Vendas Atuais": st.column_config.NumberColumn(format="%.2f"),
                    "Vendas Previstas": st.column_config.NumberColumn(format="%.2f"),
                },
                hide_index=True
            )
            
            # Download
            csv = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Sugestões em Excel/CSV",
                data=csv,
                file_name="sugestoes_otimizadas.csv",
                mime="text/csv",
            )


# --- ABA: AUDITORIA ---
elif menu == "Auditoria IA":
    st.title("🕵️ Auditoria do Modelo")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_train[FEATURES], df_train['target_log'], test_size=0.2, shuffle=False
    )
    
    # Treina validador
    validador = HistGradientBoostingRegressor(max_iter=100, monotonic_cst=MONOTONIC_CONSTRAINTS, random_state=42)
    validador.fit(X_train, y_train)
    preds = np.expm1(validador.predict(X_test))
    reais = np.expm1(y_test)
    
    r2 = r2_score(reais, preds)
    mae = mean_absolute_error(reais, preds)
    
    c1, c2 = st.columns(2)
    c1.metric("Precisão (R²)", f"{r2:.2%}", help="Quanto o modelo explica das vendas.")
    c2.metric("Erro Médio (MAE)", f"{mae:.2f} un", help="Erro médio em unidades por produto.")
    
    st.subheader("O que importa para a IA?")
    r = permutation_importance(validador, X_test, y_test, n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importancia': r.importances_mean}).sort_values('Importancia', ascending=True)
    
    fig = px.bar(imp_df, x='Importancia', y='Feature', orientation='h', title="Peso na Decisão de Compra")
    st.plotly_chart(fig)

# --- ABA: BANCO DE DADOS ---
elif menu == "Banco de Dados":
    st.title("📂 Gestão de Dados")
    
    # --- DOWNLOAD DE TEMPLATE ---
    st.markdown("### 📥 Template para Atualização")
    st.info("Baixe o template com as colunas corretas para facilitar o preenchimento dos dados.")
    
    # Cria template com colunas esperadas e exemplos
    template_data = {
        'Codigo': ['PROD001', 'PROD002', 'PROD003'],
        'B1 Desc': ['Produto Exemplo 1', 'Produto Exemplo 2', 'Produto Exemplo 3'],
        'Day of DATA': ['2024-12-01', '2024-12-01', '2024-12-01'],
        'Categoria': ['CATEGORIA A', 'CATEGORIA B', 'CATEGORIA A'],
        'custo': [50.00, 75.00, 30.00],
        'preço varejo': [100.00, 150.00, 60.00],
        'varejo concorrente': [95.00, 145.00, 65.00],
        'un.7d varejo': [50, 30, 100],
        'estoque': [500, 200, 800],
        'Custo ADS com PAI': [10.00, 15.00, 5.00],
        'tx conv aprox': [2.5, 3.0, 1.8]
    }
    
    df_template = pd.DataFrame(template_data)
    csv_template = df_template.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Baixar Template CSV",
        data=csv_template,
        file_name="template_atualizacao.csv",
        mime="text/csv",
        help="Baixe este arquivo, preencha com seus dados e faça o upload abaixo"
    )
    
    st.markdown("**Colunas do Template:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Codigo**: Código do produto (obrigatório)
        - **B1 Desc**: Nome/descrição do produto
        - **Day of DATA**: Data de referência (AAAA-MM-DD)
        - **Categoria**: Categoria do produto
        - **custo**: Custo unitário do produto
        - **preço varejo**: Preço de venda
        """)
    
    with col2:
        st.markdown("""
        - **varejo concorrente**: Preço do concorrente
        - **un.7d varejo**: Vendas dos últimos 7 dias
        - **estoque**: Estoque atual
        - **Custo ADS com PAI**: Custo de anúncios
        - **tx conv aprox**: Taxa de conversão (%)
        """)
    
    st.markdown("---")
    
    # --- UPLOAD DE ARQUIVO ---
    st.markdown("### 📤 Importar Dados")
    uploaded_file = st.file_uploader("Importar novas vendas (Excel/CSV)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        if st.button("Processar e Adicionar ao Banco"):
            with st.spinner("Limpando dados, atualizando SQLite e treinando IA..."):
                sucesso, n_novos, n_total = processar_upload(uploaded_file)
                if sucesso:
                    st.success(f"Sucesso! {n_novos} registros processados. Total no Banco: {n_total}.")
                    st.cache_resource.clear() # Limpa cache para forçar re-treino na próxima ação
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Erro: {n_novos}")
    
    st.markdown("---")
    
    # --- VISUALIZAÇÃO DOS DADOS ---
    st.markdown("### 👁️ Visualizar e Explorar Dados")
    
    if os.path.exists(ARQUIVO_MESTRE):
        try:
            df_banco = pd.read_csv(ARQUIVO_MESTRE)
            
            # Converte data_referencia para datetime se existir
            if 'data_referencia' in df_banco.columns:
                df_banco['data_referencia'] = pd.to_datetime(df_banco['data_referencia'], errors='coerce')
            
            # ESTATÍSTICAS GERAIS
            st.markdown("#### 📊 Estatísticas do Banco")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Registros", f"{len(df_banco):,}")
            
            with col2:
                produtos_unicos = df_banco['id_produto'].nunique() if 'id_produto' in df_banco.columns else 0
                st.metric("Produtos Únicos", f"{produtos_unicos:,}")
            
            with col3:
                if 'categoria' in df_banco.columns:
                    categorias_unicas = df_banco['categoria'].nunique()
                    st.metric("Categorias", f"{categorias_unicas}")
                else:
                    st.metric("Categorias", "N/A")
            
            with col4:
                if 'data_referencia' in df_banco.columns:
                    data_min = df_banco['data_referencia'].min()
                    data_max = df_banco['data_referencia'].max()
                    if pd.notna(data_min) and pd.notna(data_max):
                        dias = (data_max - data_min).days
                        st.metric("Período", f"{dias} dias")
                    else:
                        st.metric("Período", "N/A")
                else:
                    st.metric("Período", "N/A")
            
            st.markdown("---")
            
            # FILTROS
            st.markdown("#### 🔍 Filtros")
            
            col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
            
            with col_filtro1:
                # Filtro por Produto
                if 'id_produto' in df_banco.columns:
                    produtos_disponiveis = ['Todos'] + sorted(df_banco['id_produto'].astype(str).unique().tolist())
                    produto_selecionado = st.selectbox("Filtrar por Produto", produtos_disponiveis)
                else:
                    produto_selecionado = 'Todos'
            
            with col_filtro2:
                # Filtro por Categoria
                if 'categoria' in df_banco.columns:
                    categorias_disponiveis = ['Todas'] + sorted(df_banco['categoria'].dropna().unique().tolist())
                    categoria_selecionada = st.selectbox("Filtrar por Categoria", categorias_disponiveis)
                else:
                    categoria_selecionada = 'Todas'
            
            with col_filtro3:
                # Filtro por Data
                if 'data_referencia' in df_banco.columns and df_banco['data_referencia'].notna().any():
                    data_min = df_banco['data_referencia'].min()
                    data_max = df_banco['data_referencia'].max()
                    
                    if pd.notna(data_min) and pd.notna(data_max):
                        periodo_selecionado = st.date_input(
                            "Período",
                            value=(data_min.date(), data_max.date()),
                            min_value=data_min.date(),
                            max_value=data_max.date()
                        )
                    else:
                        periodo_selecionado = None
                else:
                    periodo_selecionado = None
            
            # Aplicar filtros
            df_filtrado = df_banco.copy()
            
            if produto_selecionado != 'Todos' and 'id_produto' in df_banco.columns:
                df_filtrado = df_filtrado[df_filtrado['id_produto'].astype(str) == produto_selecionado]
            
            if categoria_selecionada != 'Todas' and 'categoria' in df_banco.columns:
                df_filtrado = df_filtrado[df_filtrado['categoria'] == categoria_selecionada]
            
            if periodo_selecionado and 'data_referencia' in df_banco.columns:
                if len(periodo_selecionado) == 2:
                    inicio, fim = periodo_selecionado
                    df_filtrado = df_filtrado[
                        (df_filtrado['data_referencia'].dt.date >= inicio) & 
                        (df_filtrado['data_referencia'].dt.date <= fim)
                    ]
            
            st.info(f"📋 Exibindo {len(df_filtrado):,} de {len(df_banco):,} registros")
            
            # TABELA INTERATIVA
            st.markdown("#### 📋 Dados Filtrados")
            
            # Selecionar colunas para exibir
            colunas_disponiveis = df_filtrado.columns.tolist()
            colunas_padrao = ['id_produto', 'nome_produto', 'data_referencia', 'categoria', 
                             'preco_varejo', 'custo_unitario', 'qtd_venda_7d', 'estoque_atual']
            colunas_exibir = [col for col in colunas_padrao if col in colunas_disponiveis]
            
            # Se não tiver as colunas padrão, pega as primeiras 8
            if not colunas_exibir:
                colunas_exibir = colunas_disponiveis[:8]
            
            with st.expander("⚙️ Selecionar Colunas para Exibir"):
                colunas_selecionadas = st.multiselect(
                    "Escolha as colunas",
                    colunas_disponiveis,
                    default=colunas_exibir
                )
            
            if colunas_selecionadas:
                df_exibir = df_filtrado[colunas_selecionadas].copy()
                
                # Formata data se presente
                if 'data_referencia' in df_exibir.columns:
                    df_exibir['data_referencia'] = df_exibir['data_referencia'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(
                    df_exibir,
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
                
                # Botão para baixar dados filtrados
                csv_filtrado = df_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Dados Filtrados (CSV)",
                    data=csv_filtrado,
                    file_name="dados_filtrados.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Selecione pelo menos uma coluna para exibir.")
            
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
    else:
        st.warning("⚠️ Nenhum dado encontrado no banco. Faça o upload de um arquivo primeiro.")
    
    st.markdown("---")
    
    # --- BACKUP ---
    st.markdown("### 💾 Backup dos Dados")
    if os.path.exists(ARQUIVO_MESTRE):
        st.info(f"Base Atual: {len(pd.read_csv(ARQUIVO_MESTRE))} registros históricos.")
        with open(ARQUIVO_MESTRE, "rb") as f:

            st.download_button("📥 Baixar Backup CSV", f, file_name="backup_dados.csv")
