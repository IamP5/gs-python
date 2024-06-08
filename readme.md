# Aquamarine Python

## Alunos

- Bruno Gabriel Silva Dominicheli   RM 554981
- Larissa Rodrigues Lapa            RM 554517
- Nicolly Ramalho Bleinat           RM 555359​

## Descrição do Projeto

Aquamarine é um projeto de análise e previsão de dados da saúde dos oceanos. Ele utiliza técnicas de regressão para prever valores futuros de pH, temperatura da água e níveis de microplásticos no mar. Além disso, o projeto inclui uma interface interativa para consulta e visualização de dados históricos e previsões, contando também com uma integração com LLM para gerar respostas com contexto da base de dados e interação com o usuário.

## Instruções de Uso

### Execuição

1. Clone o repositório
2. Navegue até a pasta do projeto
3. Instale as dependencias necessárias
4. Adicione uma variável de ambiente `OPENAI_API_KEY com o valor de uma API key da open AI ou atribua a uma string vazia caso não possua`
5. Execute o script principal `python main.py`

### Menu de opções

1. Ver dados atuais de um estado: Mostra os dados históricos de um estado específico em um ano determinado.
2. Ver previsão de pH de um estado: Gera previsões de pH da água do mar para um estado até um ano final usando vários métodos de regressão.
3. Ver previsão de temperatura de um estado: Gera previsões da temperatura da água do mar para um estado até um ano final usando vários métodos de regressão.
4. Ver previsão de microplásticos de um estado: Gera previsões dos níveis de microplásticos no mar para um estado até um ano final usando vários métodos de regressão.
5. Consultar Aqua AI: Permite fazer perguntas sobre os dados ao modelo de IA integrado.
6. Sair: Encerra o programa.

## Requisitos

- Python

**Bibliotecas:**

- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- OpenAI

## Informações Relevante

### Modelos de Previsão

O projeto inclui os seguintes modelos de previsão:

- Regressão Linear
- Regressão Polinomial
- Regressão Senoidal
- Regressão Quadrática
- Regressão Logarítmica

### Funções Principais

- load_data: Carrega dados gerados aleatoriamente para os estados brasileiros.
- data_prediction: Realiza a previsão dos dados utilizando o método especificado.
- show_data: Mostra os dados históricos de um estado em um ano específico.
- show_prediction: Gera e exibe as previsões de dados para um estado até um ano final.
- call_aqua_ai: Interage com a API da OpenAI para responder perguntas sobre os dados.
