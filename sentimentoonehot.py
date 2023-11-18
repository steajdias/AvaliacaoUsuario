import pandas as pd
import re
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import CountVectorizer

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

# Lista de stopwords comuns em português
stopwords = ["o", "a", "os", "as", "e", "é", "um", "mas", "uma", "até", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "como", "quando"]

# Frase para avaliar
frase_para_avaliar = """"Embora tenha considerado desistir do estudo em Processamento de Linguagem Natural (PLN) devido à sua complexidade,
            a experiência transformadora do curso ministrado por Stephanie me surpreendeu positivamente. O conteúdo desafiador foi apresentado de
            maneira envolvente e acessível, proporcionando não apenas um entendimento mais profundo dos temas intricados, mas também despertando um
            entusiasmo genuíno pelo campo. O curso não apenas superou minhas expectativas, mas também revelou a fascinante interseção entre teoria e
            prática, tornando o aprendizado não apenas educativo, mas também incrivelmente divertido. Quem poderia imaginar que a exploração do PLN se
            tornaria uma jornada tão estimulante e recompensadora?"""

# Tokenizar a frase e remover stopwords personalizadas
palavras_tokenizadas = [palavra.lower() for palavra in re.findall(r'\b\w+\b', frase_para_avaliar) if palavra.lower() not in stopwords]

# Criar um dicionário com palavras únicas e atribuir um índice para cada palavra
vocabulario = {palavra: idx for idx, palavra in enumerate(set(palavras_tokenizadas))}

# Criar um vetor one-hot para a frase
vetor_one_hot = [1 if palavra in palavras_tokenizadas else 0 for palavra in vocabulario]

# Imprimir informações sobre cada palavra na frase
print("\nInformações sobre cada palavra na frase:")
for palavra, idx in vocabulario.items():
    polaridade = df[df['Lemma'].str.lower() == palavra]['Polarity'].iloc[0] if not df[df['Lemma'].str.lower() == palavra].empty else 0
    print(f"Palavra: '{palavra}', Índice: {idx}, Polaridade: {polaridade}")

# Calcular a polaridade da frase com base nas polaridades individuais das palavras
polaridades_individuais = [df[df['Lemma'].str.lower() == palavra]['Polarity'].iloc[0] if not df[df['Lemma'].str.lower() == palavra].empty else 0 for palavra in palavras_tokenizadas]
polaridade_frase = sum(polaridades_individuais)

# Determinar a polaridade da frase
if polaridade_frase > 0:
    polaridade_frase = 1
elif polaridade_frase < 0:
    polaridade_frase = -1
else:
    polaridade_frase = 0

# Imprimir o resultado da previsão da frase
print(f"\nPolaridade da Frase: {polaridade_frase}")


# Criar um vetor one-hot para a frase usando CountVectorizer
vectorizer = CountVectorizer(vocabulary=vocabulario, binary=True)
matriz_one_hot = vectorizer.fit_transform([' '.join(palavras_tokenizadas)])

# Obter o cabeçalho das palavras
palavras_header = list(vocabulario.keys())

# Criar uma matriz esparsa com zeros e uns
matriz_one_hot_sparse = lil_matrix((len(palavras_header), len(palavras_header)), dtype=int)
matriz_one_hot_sparse.setdiag(1)

# Criar um DataFrame para imprimir com cabeçalho
df_matriz_one_hot = pd.DataFrame(matriz_one_hot_sparse.toarray(), columns=palavras_header, index=palavras_header)

# Imprimir a matriz one-hot esparsa com cabeçalho e diagonal 1
print("\nMatriz One-Hot Esparsa:")
print(df_matriz_one_hot)

