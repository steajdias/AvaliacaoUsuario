import pandas as pd
import re

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
            a experiência transformadora da forma como a Stephanie explica me surpreendeu positivamente. O conteúdo desafiador foi apresentado de
            maneira envolvente e acessível, proporcionando não apenas um entendimento mais profundo dos temas intricados, mas também despertando um
            entusiasmo genuíno pelo campo. O curso não apenas superou minhas expectativas, mas também revelou a fascinante interseção entre teoria e
            prática, tornando o aprendizado não apenas educativo, mas também incrivelmente divertido. Quem poderia imaginar que a exploração do PLN se
            tornaria uma jornada tão estimulante e recompensadora?"""

# Tokenizar a frase e remover stopwords personalizadas
palavras_tokenizadas = [palavra.lower() for palavra in re.findall(r'\b\w+\b', frase_para_avaliar) if palavra.lower() not in stopwords]

# Imprimir informações sobre cada palavra na frase
print("\nInformações sobre cada palavra na frase:")
for palavra in palavras_tokenizadas:
    idx = df[df['Lemma'].str.lower() == palavra].index[0] if not df[df['Lemma'].str.lower() == palavra].empty else -1
    polaridade = df.loc[idx, 'Polarity'] if idx != -1 else 0
    print(f"Palavra: '{palavra}', Polaridade: {polaridade}")

# Verificar se há pelo menos uma palavra negativa na frase
tem_palavra_negativa = any(df[df['Lemma'].str.lower().isin(palavras_tokenizadas)]['Polarity'].eq(-1))

# Determinar a polaridade da frase
polaridade_frase = -1 if tem_palavra_negativa else 1

# Imprimir o resultado da previsão da frase
print(f"\nPolaridade da Frase: {polaridade_frase}")

# Criar um vetor one-hot para a frase usando CountVectorizer
palavras_tokenizadas = [palavra.lower() for palavra in re.findall(r'\b\w+\b', frase_para_avaliar) if palavra.lower() not in stopwords]
vectorizer = CountVectorizer(vocabulary=set(palavras_tokenizadas), binary=True)
matriz_one_hot = vectorizer.fit_transform([' '.join(palavras_tokenizadas)])

# Obter o cabeçalho das palavras
palavras_header = vectorizer.get_feature_names_out()

# Criar uma matriz esparsa com zeros e uns
matriz_one_hot_sparse = lil_matrix((len(palavras_header), len(palavras_header)), dtype=int)
matriz_one_hot_sparse.setdiag(1)

# Criar um DataFrame para imprimir com cabeçalho
df_matriz_one_hot = pd.DataFrame(matriz_one_hot_sparse.toarray(), columns=palavras_header, index=palavras_header)

# Imprimir a matriz one-hot esparsa com cabeçalho e diagonal 1
print("\nMatriz One-Hot Esparsa:")
print(df_matriz_one_hot)
