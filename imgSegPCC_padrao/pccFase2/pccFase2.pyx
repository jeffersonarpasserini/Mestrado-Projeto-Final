import random
import numpy as np
import collections


def pccFase2(container):
    #laco principal do algoritmo
    
    #variaveis de controle
    maxMedDominio = np.double(0)
    controleParada = 0

    #Carrega dados parametros
    #[dominio atual, no rotulado]
    arrNodes = container[0]

    if (len(arrNodes) == 0):
        print("Erro array vertices...")
    else:
        print("Nro Vertices :"+str(len(arrNodes)))
        
    #[class 0, class 1, class 2, ..., class n]
    arrDominio = container[1]

    if (len(arrDominio) == 0):
        print("Erro array dominios...")
    else:
        print("Nro dominios :" + str(len(arrDominio)))

    #[no, nViz] --> armazena o no vizinho
    arrNeig = container[2]
    if (len(arrNeig) == 0):
        print("Erro array vizinhos...")
    else:
        print("Nro Vizinhos:" + str(len(arrNeig)))
        
    #[pos atual, forca, no casa, classe]
    arrParticle = container[3]
    if (len(arrParticle) == 0):
        print("Erro array particulas...")
    else:
        print("Nro Particulas :" + str(len(arrParticle)))
        
    #[part, no] --> armazena a distancia
    arrDistancia = container[4]
    if (len(arrDistancia) == 0):
        print("Erro array distancias...")
    else:
        print("Distancias :" + str(len(arrDistancia)))
        
    #maximo de iteracoes do algoritmo
    maxIte = container[5]
    #ponto de parada controle
    maxParada = container[6]
    #probabilidade mov aleatorio ou guloso
    pGrd = container[7]
    #red domino do no pela partiula
    deltaV = container[8]
    #nro de classe
    nClasses = container[9]
    #fator controle de parada
    fatorControleParada = container[10]
    #imprime na tela status do processamento
    imprime = container[11]

    continua = True
    nIte = 0 

    while ((nIte < maxIte) and continua):
        if (imprime and (nIte%10==0)):
            print("Iteracao:" + str(nIte) + " -- Controle Parada:" + str(controleParada) + " -- maxDominio: " + str(maxMedDominio))
        #executa para cada particula
        for nroPart in range(0,len(arrParticle)):
            #if (imprime):
            #   print("Iteracao:" + str(nIte) + " -- Part:" + str(nroPart) + " -- Parada:" + str(controleParada)+ " -- Dominio:" + str(maxMedDominio))
            #dados da particula
            posAtualPart = np.int64(arrParticle[nroPart,0])
            forcaPart = arrParticle[nroPart,1]
            casaPart = np.int64(arrParticle[nroPart,2])
            classePart = np.int32(arrParticle[nroPart,3])
            distAtualPart = arrDistancia[nroPart,posAtualPart]

            #sorteia um numero aleatorio para escolha do movimento
            r = random.uniform(0,1)

            #busca vertices vizinho a posicao atual da particula
            aVizinhos = arrNeig[posAtualPart]

            #variavel armazena vizinho escolhido
            vizinho = -1
            #escolhe entre movimento guloso ou aleatorio
            if (r < pGrd):
                #movimento guloso

                #variavel soma probabilidades
                somaProb = np.double(0)
                #calcula somatorio da relacao distancia e forca dos vizinhos
                #para a classe da particula
                for nv in range(0,len(aVizinhos)):
                    #seleciona o vizinho do vetor
                    viz = aVizinhos[nv]
                    #pega a forca do vizinho
                    forcaNo = arrDominio[viz,classePart]
                    #seleciona a distancia do no vizinho
                    distCasa = arrDistancia[nroPart,viz]
                    somaProb = somaProb + (distCasa*(1/pow((1+forcaNo),-2)))

                #calcula a roleta de probabilidade de escolha de cada vizinho
                prob = np.double(0)
                probsum = np.ndarray(shape=(len(aVizinhos),4),dtype=np.double)
                anterior = -0.1
                atual = 0
                for nv in range(0,len(aVizinhos)):
                    #seleciona o vizinho do vetor
                    viz = aVizinhos[nv]
                    #pega a forca do vizinho
                    forcaNo = arrDominio[viz,classePart]
                    #seleciona a distancia do vizinho
                    distCasa = arrDistancia[nroPart,viz]
                    #calcula a probabilidade do vizinho selecionado
                    prob = (distCasa*(1/pow((1+forcaNo),-2)))/somaProb
                    atual = atual+prob
                    #cria faixa de probabilidade para a roleta entre 0 e 1
                    probsum[nv,0] = viz
                    probsum[nv,1] = prob
                    probsum[nv,2] = anterior
                    probsum[nv,3] = atual
                    anterior=atual

                #realiza o sorteio ponderado
                randPond = np.double(random.uniform(0,1))

                #define o vizinho a ser visitado
                for nv in range(0,len(probsum)):
                    if(randPond>probsum[nv,2] and randPond<=probsum[nv,3]):
                        vizinho = np.int32(probsum[nv,0])

            else:
                #movimento aleatorio
                vizinho = random.choice(aVizinhos)

            #atualiza a tabela de distancias
            
            #pega a distancia do vizinho selecionado
            distVizinho = arrDistancia[nroPart,vizinho]
            #se a dist atual do viz for maior que o calculado -->atualiza
            if (distVizinho > (distAtualPart+1)):
                arrDistancia[nroPart,vizinho] = distAtualPart+1

            #calcula a nova relacao de dominio para o vertice visitado
        
            #aplica redutor a forca da particula
            redForcaPart = np.double((deltaV*forcaPart)/(nClasses-1))
            acumForca = np.double(0)
            
            #para cada classe do vertice vizinho selecionado
            #calcula a reducao de dominio sobre o vertice para
            #classes diferentes da particula
            
            rotulado = np.int32(arrNodes[vizinho,1])

            for nc in range(0,nClasses):
                if(nc != classePart):
                    #dominio atual menos a fracao da forca da part
                    dominioClasse = np.double(arrDominio[vizinho,nc])
                    dominioClasse = np.double(dominioClasse-redForcaPart)
                    #se dominio menor que zero iguala a zero
                    if(dominioClasse<0):
                        dominioClasse=np.double(0)
                        #acumula o restante da forca da classe do vertice visitado
                        acumForca = np.double(acumForca)+np.double(arrDominio[vizinho,nc])
                    else:
                        #acumula a reducao realizada pela particula
                        #soma a forca da particula reduzida
                        acumForca = np.double(acumForca) + np.double(redForcaPart)

                    #se o vertice nao foi rotulado pelo especialista
                    #atualiza relacao de dominio  
                    if (rotulado == 0):
                        #atualiza a forca no vizinho selecionado na classe atual
                        arrDominio[vizinho,nc] = np.double(dominioClasse)

            #se o vertice nao foi rotulado pelo especialista
            #atualiza a relacao de dominio
            if (rotulado == 0):
                #atualiza o dominio do vertice visitado para a classe da particula
                #Esta relacao e dada de 1-acumulado das reducoes das outras classes
                arrDominio[vizinho,classePart] = np.double(arrDominio[vizinho,classePart]) + np.double(acumForca)

            #atualiza a forca da particula
            arrParticle[nroPart,1] = arrDominio[vizinho,classePart]

            #verifica classe dominante
            #se o vertice nao foi rotulado pelo especialista
            if (rotulado == 0):
                classeDominante = -1
                vlrDominante = -1
                #extrai da matriz a linha de dominio do vizinho selecionado
                rowDominio = arrDominio[vizinho]
                #determina o maior valor para o dominio do vizinho
                vlrDominante = max(rowDominio) #pega valor maximo do dominio do vizinho
                classeDominante = np.int32(np.argmax(rowDominio)) #posicao do valor maximo de dominio representa classe
                #verifica se existe valor de dominio igual
                if(any(vlrDominante == vlr for vlr in rowDominio)):
                    antes = arrNodes[vizinho,0]
                    arrNodes[vizinho,0] = classeDominante
                    #print("Trocou " + str(vizinho) + " para --> " + str(arrNodes[vizinho,0]) + " de: " + str(antes))

            #verifica se a particula obteve dominio sob o no vizinho
            if (classePart == classeDominante):
                #atualiza posicao atual da particula
                arrParticle[nroPart,0] = vizinho


            dominioMedio = np.double(0.00)
            maiorDominio = np.double(-1.00)
            #verifica criterio de parada
            if (nIte%10==0):
                #gera um vetor com o maior valor de cada vertice representado na matriz
                maioresDominios = np.amax(arrDominio, axis=1)
                #media dos maiores dominios
                dominioMedio = np.average(maioresDominios)

                if ( (dominioMedio-maxMedDominio) > fatorControleParada):
                    maxMedDominio = dominioMedio
                    controleParada = 0
                else:
                    controleParada=controleParada+1
                    if(controleParada>maxParada):
                        continua=False

        nIte = nIte+1
    
    #gera retorno da fase2
    retFase2={}
    retFase2.has_key(0)
    retFase2[0] = arrNodes
    retFase2.has_key(1)
    retFase2[1] = arrDominio

    return retFase2



































