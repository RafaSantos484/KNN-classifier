# Sobre o Projeto

Este projeto visa gerar um classificador de imagens usando o algoritmo KNN(K-Nearest Neighbors).

# Sobre o Algoritmo KNN

Dado um conjunto de vetores multidimensionais e suas respectivas classes como input, a fase de treinamento deste algoritmo consiste apenas em armazenar o vetor de features e classes deste input.

Na fase de classificação, a classe de um input será definida pela classe mais recorrente dos `K` vizinhos mais próximos, sendo `K` um parâmetro definido pelo usuário. Pode-se também definir como será medida a distância entre dois elementos. Alguns métricas comuns para inputs numéricos são a distância euclidiana e de Manhattan.

![1726496902281](image/README/1726496902281.png)

![1726497234424](image/README/1726497234424.png)

![1726497263796](image/README/1726497263796.png)

No caso deste projeto, as features das imagens usadas são os valores dos seus pixels. O treinamento foi feito usando a biblioteca [sklearn](https://scikit-learn.org/stable/).
