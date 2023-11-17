# Importar bibliotecas necessárias
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Leitura do arquivo com uma base de dados que diz quais palavras podem ter uma polaridade negativa ou positiva
file_path = '/content/SentiLex-lem-PT02.txt'
entries = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            lemma, info = line.strip().split('.', 1)
            entry_parts = info.split(';')
            pos = entry_parts[0].split('=')[1]
            target = entry_parts[1].split('=')[1]
            polarity = entry_parts[2].split('=')[1]
            annotation = entry_parts[3].split('=')[1]
            entries.append((lemma, pos, target, int(polarity), annotation))
        except ValueError:
            pass

# Criar DataFrame
df = pd.DataFrame(entries, columns=['Lemma', 'POS', 'Target', 'Polarity', 'Annotation'])

# Filtrar apenas as palavras com polaridade positiva ou negativa
df_filtered = df[(df['Polarity'] == 1) | (df['Polarity'] == -1)]

# Importar bibliotecas necessárias
from sklearn.feature_extraction.text import CountVectorizer

# Criar uma representação one-hot encoding para as palavras
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df_filtered['Lemma'])

# Adicionar coluna 'Polarity' ao DataFrame one-hot
df_one_hot = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_one_hot['Polarity'] = df_filtered['Polarity']

# Exibir as primeiras linhas da matriz one-hot
print("Matriz One-Hot Head:")
print(df_one_hot.head())

import numpy as np

# ...

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_one_hot.drop('Polarity', axis=1), df_one_hot['Polarity'], test_size=0.2, random_state=42)

# Garantir que y_train tenha o mesmo conjunto de índices que X_train
y_train = df_one_hot.loc[X_train.index, 'Polarity']

# Verificar e remover valores nulos em X_train e y_train
X_train = X_train.dropna()

y_train = y_train.loc[X_train.index]

df_combined = pd.concat([X_train, y_train], axis=1)
df_combined = df_combined.dropna()
X_train = df_combined.drop('Polarity', axis=1)
y_train = df_combined['Polarity']

X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

# Fazer previsões para palavras individuais
palavras_para_avaliar = ["ótimo", "ruim"]

# Transformar as palavras para avaliar usando o mesmo vetorizador usado no treinamento
palavras_one_hot = vectorizer.transform(palavras_para_avaliar)

# Converter a representação esparsa para uma matriz densa
palavras_one_hot_dense = palavras_one_hot.toarray()

# Imprimir informações sobre cada palavra
for i, palavra in enumerate(palavras_para_avaliar):
    palavra_one_hot_i = palavras_one_hot_dense[i]
    # Reshape para garantir que tenha a mesma forma que o treinamento
    previsao = model.predict(palavra_one_hot_i.reshape(1, -1))
    print(f"Palavra: '{palavra}' - Previsão: {previsao[0]}")

# ...

# Criar sua própria lista de stopwords personalizada
minhas_stopwords = ['a', 'à', 'ao','da', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as']

# Adicionar mais palavras se necessário

# Fazer previsões em um conjunto de frases
frases_para_avaliar = ["O curso da Stephanie é ótimo"]
frases_para_avaliar_one_hot = vectorizer.transform(frases_para_avaliar)

# Converter a representação esparsa para uma matriz densa
frases_para_avaliar_dense = frases_para_avaliar_one_hot.toarray()

import re

def avaliar_frases(frases, vectorizer, model, stopwords=[]):
    for i, frase in enumerate(frases):
        print(f"Frase: '{frase}'")

        # Tokenizar a frase e remover stopwords personalizadas
        palavras_tokenizadas = [palavra.lower() for palavra in re.findall(r'\b\w+\b', frase) if palavra.lower() not in stopwords]

        # Inicializar rótulo da frase como positivo
        rotulo_frase = 'positiva'

        # Verificar cada palavra na frase
        for palavra in palavras_tokenizadas:
            # Verificar se a palavra está na base de dados
            if palavra in vectorizer.get_feature_names_out():
                # Obter a representação one-hot da palavra
                palavra_one_hot = vectorizer.transform([palavra])

                # Converter a representação esparsa para uma matriz densa
                palavra_one_hot_dense = palavra_one_hot.toarray()

                # Fazer a previsão usando o modelo SVM treinado
                previsao = model.predict(palavra_one_hot_dense)

                # Se a previsão for negativa, atualizar o rótulo da frase
                if previsao[0] == -1:
                    rotulo_frase = 'negativa'

                # Imprimir informações sobre a palavra
                info_palavra = df[df['Lemma'] == palavra]
                print(f"Palavra: '{palavra}' - Informações: {info_palavra.to_dict(orient='records')}, Previsão: {previsao[0]}")
            else:
                print(f"Palavra: '{palavra}' não encontrada na base de dados.")

        # Imprimir o rótulo final da frase
        print(f"Rótulo da Frase: {rotulo_frase}")
        print()

# Exemplo 1 de uso da função
frases_para_avaliar = ["O curso da Stephanie é ótimo"]
avaliar_frases(frases_para_avaliar, vectorizer, model, stopwords=minhas_stopwords)

# Exemplo 2 de uso da função
frases_para_avaliar = ["Mas meu humor hoje está péssimo"]
avaliar_frases(frases_para_avaliar, vectorizer, model, stopwords=minhas_stopwords)
