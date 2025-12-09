# Pricing 2.0 AI 💸

Sistema inteligente de otimização de preços usando Machine Learning.

## 🚀 Funcionalidades

- **Otimização de Preços**: IA analisa histórico e sugere preços ideais
- **Análise de Impacto**: Visualize ganhos potenciais por categoria e curva ABC
- **Auditoria IA**: Veja a importância de cada fator nas decisões
- **Gestão de Dados**: Upload, visualização e filtragem de dados históricos
- **Sistema de Login**: Autenticação segura com hash SHA-256

## 📋 Pré-requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

## 🔧 Instalação Local

```bash
# Clone o repositório
git clone https://github.com/SEU_USUARIO/pricing.git
cd pricing

# Instale as dependências
pip install -r requirements.txt

# Execute o aplicativo
streamlit run app.py
```

## 🔐 Login Padrão

- **Usuário**: admin
- **Senha**: admin

⚠️ **IMPORTANTE**: Altere a senha padrão em produção!

## 📁 Estrutura do Projeto

```
pricing/
├── app.py                    # Aplicação principal
├── requirements.txt          # Dependências Python
├── usuarios.json            # Credenciais (não commitado)
├── .gitignore               # Arquivos ignorados
└── README.md                # Este arquivo
```

## 🌐 Deploy no Streamlit Cloud

1. Faça upload do código no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório
4. Configure o arquivo `usuarios.json` nos **Secrets**
5. Deploy!

## 📊 Como Usar

1. Faça login com as credenciais
2. Vá em **Banco de Dados** e faça upload dos seus dados
3. Em **Otimização**, gere sugestões de preços
4. Analise o impacto por categoria e ABC
5. Baixe as sugestões em CSV

## 🛡️ Segurança

- Senhas armazenadas com hash SHA-256
- Autenticação obrigatória
- Arquivo `usuarios.json` no .gitignore

## 📝 Licença

Este projeto é privado e confidencial.

