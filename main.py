import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from openai import OpenAI

openai_client = OpenAI()
chat_history = []

data = {}

STATES = {
    "Amapá": "AP", "Bahia": "BA", "Ceará": "CE", "Espírito Santo": "ES",
    "Maranhão": "MA", "Pará": "PA", "Paraíba": "PB", "Pernambuco": "PE",
    "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "Santa Catarina": "SC", "São Paulo": "SP",
    "Sergipe": "SE", "Alagoas": "AL"
}
YEARS = np.arange(1980, 2024)
PREDICTION_MODELS = ["linear", "polynomial", "sine", "quadratic", "logarithmic", "random_forest"]

def generate_random_data(years, low, high, decimals):
    return [round(value, decimals) for value in np.random.uniform(low, high, len(years))]

def load_data():
    for state, sigla in STATES.items():
        ph_values = generate_random_data(YEARS, 7.9, 8.3, 2)
        temperatures = generate_random_data(YEARS, 20, 27, 1)
        microplastic_levels = generate_random_data(YEARS, 0.1, 2.0, 2)

        state_data = {
            year: {"ph": ph_values[i], "temperatura": temperatures[i], "microplasticos": microplastic_levels[i]}
            for i, year in enumerate(YEARS)
        }
        data[state] = {"sigla": sigla, "dados": state_data}

def fit_model(x, y, model):
    x = np.array(x).reshape(-1, 1)
    model.fit(x, y)
    return model

def linear_regression_prediction(x, y, future_years):
    model = fit_model(x, y, LinearRegression())
    return model.predict(np.array(future_years).reshape(-1, 1)).tolist()

def polynomial_regression_prediction(x, y, future_years, degree):
    model = fit_model(x, y, make_pipeline(PolynomialFeatures(degree), LinearRegression()))
    return model.predict(np.array(future_years).reshape(-1, 1)).tolist()

def sine_function(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def sine_regression_prediction(x, y, future_years):
    popt, _ = curve_fit(sine_function, np.array(x), y, maxfev=10000)
    return sine_function(np.array(future_years), *popt).tolist()

def quadratic_regression_prediction(x, y, future_years):
    return polynomial_regression_prediction(x, y, future_years, degree=2)

def logarithmic_function(x, a, b):
    return a * np.log(x) + b

def logarithmic_regression_prediction(x, y, future_years):
    popt, _ = curve_fit(logarithmic_function, np.array(x), y, maxfev=10000)
    return logarithmic_function(np.array(future_years), *popt).tolist()

def random_forest_prediction(x, y, future_years):
    model = fit_model(x, y, RandomForestRegressor())
    return model.predict(np.array(future_years).reshape(-1, 1)).tolist()

def data_prediction(estado, tipo_dado, ano_final, metodo="linear", degree=None):
    anos = list(data[estado]['dados'].keys())
    valores = [data[estado]['dados'][ano][tipo_dado] for ano in anos]
    anos_futuros = np.arange(anos[-1] + 1, ano_final + 1).tolist()
    
    prediction_functions = {
        "linear": linear_regression_prediction,
        "polynomial": lambda x, y, fy: polynomial_regression_prediction(x, y, fy, degree),
        "sine": sine_regression_prediction,
        "quadratic": quadratic_regression_prediction,
        "logarithmic": logarithmic_regression_prediction,
        "random_forest": random_forest_prediction
    }

    return anos_futuros, prediction_functions[metodo](anos, valores, anos_futuros)

def plot_predictions(anos, valores, anos_futuros, previsoes, titulo, tipo_dado):
    plt.figure(figsize=(10, 6))
    plt.plot(anos, valores, label="Dados Passados", marker='o')
    for metodo, previsao in previsoes.items():
        plt.plot(anos_futuros, previsao, label=f"Previsão {metodo}")
    plt.xlabel("Ano")
    plt.ylabel(tipo_dado.capitalize())
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()

def show_data(estado, ano):
    if estado in data and ano in data[estado]['dados']:
        info = data[estado]['dados'][ano]
        print(f"\n{'='*40}\nO estado escolhido foi {estado} ({data[estado]['sigla']}).\nAno: {ano}")
        print(f"Valores de pH da água do mar: {info['ph']}\nTemperatura da água do mar: {info['temperatura']}°C")
        print(f"Níveis de microplásticos: {info['microplasticos']} mg/L\n{'='*40}\n")
    else:
        print("Estado ou ano não encontrado.\n" + "="*40 + "\n")

def show_prediction(estado, tipo_dado, ano_final, metodos, degree=None):
    if estado in data:
        anos = list(data[estado]['dados'].keys())
        valores = [data[estado]['dados'][ano][tipo_dado] for ano in anos]
        anos_futuros = np.arange(anos[-1] + 1, ano_final + 1).tolist()
        
        previsoes = {
            metodo: data_prediction(estado, tipo_dado, ano_final, metodo, degree)[1]
            for metodo in metodos
        }
        
        plot_predictions(anos, valores, anos_futuros, previsoes, f"Previsão para {estado} ({data[estado]['sigla']})", tipo_dado)
    else:
        print("Estado não encontrado.\n" + "="*40 + "\n")

def call_aqua_ai(openai_client, pergunta):
    messages = [
        {"role": "system", "content": f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nChat history: {chat_history}\nContext: {data}\nQuestion: \n"},
        {"role": "user", "content": pergunta}
    ]

    completion = openai_client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
  
    response_content = ""
    for chunk in completion:
        if chunk.choices[0].finish_reason == "stop":
            break
        content = chunk.choices[0].delta.content
        response_content += content
        yield content
    
    chat_history.append({"role": "user", "content": pergunta})
    chat_history.append({"role": "assistant", "content": response_content})

def menu():
    while True:
        print("\n" + "="*40)
        print("           Bem vindo ao Aquamarine!")
        print("="*40)
        print("1. Ver dados atuais de um estado")
        print("2. Ver previsão de pH de um estado")
        print("3. Ver previsão de temperatura de um estado")
        print("4. Ver previsão de microplásticos de um estado")
        print("5. Consultar Aqua AI")
        print("6. Sair")
        print("="*40)
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            estado = input("Digite o estado brasileiro: ")
            ano = int(input("Digite o ano: "))
            show_data(estado, ano)
        elif opcao in ['2', '3', '4']:
            estado = input("Digite o estado brasileiro: ")
            ano_final = int(input("Digite o ano final para previsão: "))
            metodos_input = input("Escolha os métodos de previsão separados por vírgula (linear,polynomial,sine,quadratic,logarithmic,random_forest) ou 'all' para todos: ")
            if metodos_input.lower() == 'all':
                metodos = PREDICTION_MODELS
            else:
                metodos = metodos_input.split(',')
            degree = int(input("Digite o grau do polinômio (apenas para polynomial): ")) if "polynomial" in metodos else None
            tipo_dado = "ph" if opcao == '2' else "temperatura" if opcao == '3' else "microplasticos"
            show_prediction(estado, tipo_dado, ano_final, metodos, degree)
        elif opcao == '5':
            while True:
                pergunta = input("\n\nDigite sua pergunta para o Aqua AI (ou '0' para voltar ao menu): ")
                if pergunta == '0':
                    break
                else:
                    print("\n" + "="*40)
                    print("Resposta do Aqua AI:")
                    for tokens in call_aqua_ai(openai_client, pergunta):
                        print(tokens, end="")
        elif opcao == '6':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    load_data()
    menu()
