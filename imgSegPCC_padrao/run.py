from collections import defaultdict
from datetime import datetime
from PIL import Image
import numpy as np
import scipy.misc
import sys
import time
import pccFase1 as pccF1
import pccFase2 as pccF2
import geraImagem as geraImg
import string
import csv


def executa(container):

    diretorio = "../baseImagens/"
    nomeImg = diretorio+container[0]
    nomeRotulo = diretorio+container[1]
    nomeGabarito = diretorio+container[2]
    print(nomeImg + " - " + nomeRotulo + " - " + nomeGabarito)

    #-------------------- Parametros ----------------------------------------------------------#
    #Fase 1
    verificaLigacao = False #se verdadeiro nao liga nos com rotulos de classes diferentes
    k = 200 #numero de vizinho
    nClasses = 2 #numero de classes
    featLength = [1,1,1,1,1,1,1,1,1] #pesos caracteristicas
    weigths = False #utiliza pesos

    #Fase 2
    maxIte = 1000000 #numero maximo de iteracoes
    maxParada = 15000 #controle de parada se alteracoes nos dados
    fatorControleParada = 0.001 #define a diferenca minima para incrementar o controle de parada
    Pgrd = np.double(0.5) #varivel de controle do sorteio de mov guloso ou aleatorio
    deltaV = np.double(0.1) #redutor do controle sobre o no
    imprime = True #libera status de andamento durante o processamento do algoritmo


    #-------------------  Leitura e conversao de imagens --------------------------------------#
    nowInicio = datetime.now()
    
    #imagem principal
    img = Image.open(nomeImg).convert('RGB')
    #interpolacao bicubica da imagem
    imgReduzida = scipy.misc.imresize(img,.3,interp='bicubic')
   
    arrImg = np.array(img,dtype=np.uint8)
    imgWidth = img.width
    imgHeight = img.height
    imgChannels = 3

    nRows = imgReduzida.shape[0]
    nCols = imgReduzida.shape[1]

    #escala de cinza reduzida
    grayReduzida = img.convert('L')
    arrGray = np.array(grayReduzida,dtype=np.uint8)

    #imagem com marcacao de rotulos
    imgMask = Image.open(nomeRotulo)
    imgMask = imgMask.convert('RGBA')
    #interplacao bilinear da mascara
    imgMaskReduzida = scipy.misc.imresize(imgMask,.3,interp='bicubic')
    #imgMaskReduzida = imgMaskReduzida.convert('RGB')
    #arrMask = np.array(imgMaskReduzida, dtype=np.uint8)

    #imagem gabarito para conferencia do resultado
    imgGab = Image.open(nomeGabarito)
    imgRGB = imgGab.convert('RGB')
    arrGab = np.array(imgRGB, dtype=np.uint8)
    
    tempoInicio = time.time()
    
    #prepara dados para a fase 1
    container={}
    container.has_key(0)
    container[0] = imgReduzida
    container.has_key(1)
    container[1] = imgMaskReduzida
    container.has_key(2)
    container[2] = nClasses
    container.has_key(3)
    container[3] = k
    container.has_key(4)
    container[4] = verificaLigacao
    container.has_key(5)
    container[5] = featLength
    container.has_key(6)
    container[6] = nRows
    container.has_key(7)
    container[7] = nCols
    container.has_key(8)
    container[8] = weigths
    container.has_key(9)
    container[9] = arrGray

    retorno = pccF1.pccFase1(container)

    tempoF1 = time.time()
    tempoFase1 = tempoF1-tempoInicio

    #prepara dados para fase 2
    retorno.has_key(5)
    retorno[5] = maxIte
    retorno.has_key(6)
    retorno[6] = maxParada
    retorno.has_key(7)
    retorno[7] = Pgrd
    retorno.has_key(8)
    retorno[8] = deltaV
    retorno.has_key(9)
    retorno[9] = nClasses
    retorno.has_key(10)
    retorno[10] = fatorControleParada
    retorno.has_key(11)
    retorno[11] = imprime
    
    arrRetorno = pccF2.pccFase2(retorno)

    arrNodes = arrRetorno[0]
    arrDominio = arrRetorno[1]

    nowTerminio = datetime.now()

    tempoF2 = time.time()
    tempoFase2 = tempoF2-tempoF1
    tempoFinal = tempoF2-tempoInicio

    respFinal ={}
    respFinal.has_key(0)
    respFinal[0] = arrNodes
    respFinal.has_key(1)
    respFinal[1] = arrGab
    respFinal.has_key(2)
    respFinal[2] = arrImg
    respFinal.has_key(3)
    respFinal[3] = imgWidth
    respFinal.has_key(4)
    respFinal[4] = imgHeight
    respFinal.has_key(5)
    respFinal[5] = imgChannels
    respFinal.has_key(6)
    respFinal[6] = nRows
    respFinal.has_key(7)
    respFinal[7] = nCols
    respFinal.has_key(8)
    respFinal[8] = nClasses
    respFinal.has_key(9)
    respFinal[9] = arrDominio

    respGeraImagem = geraImg.gerar(respFinal)

    respGeraImagem.has_key(10)
    respGeraImagem[10] = tempoFase1
    respGeraImagem.has_key(11)
    respGeraImagem[11] = tempoFase2
    respGeraImagem.has_key(12)
    respGeraImagem[12] = tempoFinal

    print("Data e Hora")
    print(nowInicio.hour) 
    print(nowInicio.minute)
    print(nowInicio.second)
    print("---------------------")
    print(nowTerminio.hour)
    print(nowTerminio.minute)
    print(nowTerminio.second)


    #print("------------------------------------------------------------------------------------------------")
    #print("Nro pixel fundo :" + str(contaPxFundo))
    #print("Nro Erros fundo :" + str(contaErroFundo))
    #print("Nro Acertos Fundo :" + str(contaAcertoFundo))
    #print("------------------------------------------------------------------------------------------------")
    #print("Nro pixel objeto :" + str(contaPxFrente))
    #print("Nro Erros objeto :" + str(contaErroFrente))
    #print("Nro Acertos objeto :" + str(contaAcertoFrente))
    #print("------------------------------------------------------------------------------------------------")
    #print("Nro pixel imagem :" + str(contaPxImg))
    #print("Nro Acertos :" + str(contaAcertoFundo+contaAcertoFrente))
    #print("Nro Erros : " + str(contaErroFrente+contaErroFundo))
    #print("------------------------------------------------------------------------------------------------")
    #print("Tempo de Execucao -Fase1:" + str(tempoFase1))
    #print("Tempo de Execucao -Fase2:" + str(tempoFase2))
    #print("Tempo de Execucao -Final:" + str(tempoTotal))
    #print("------------------------------------------------------------------------------------------------")

    #arq = "vertices.csv"
    #arqVertice = csv.writer(open(arq,"wb"),delimiter=";")
    #arqVertice.writerow(["Dominio", "Rotulado","Cl_0", "Cl_1"])
    #for x in range(0,len(arrNodes)):
    #    arqVertice.writerow([str(arrNodes[x,0]),str(arrNodes[x,1]),str(arrDominio[x,0]),str(arrDominio[x,1])])

    return respGeraImagem

def main():

    nVezes = 30
    #nomeImg      = ["train_2007_002462.png"     , "train_2007_003286.png"     ]
    #nomeRotulo   = ["train_2007_002462_anno.png", "train_2007_003286_anno.png"]
    #nomeGabarito = ["train_2007_002462_GT.png"  , "train_2007_003286_GT.png"  ]

    nomeImg =["GT13.png"]
    nomeRotulo=["GT13_anno.png"]
    nomeGabarito=["GT13_GT.png"]

    for x in range(0, len(nomeImg)):
        parametro = {}
        parametro.has_key(0)
        parametro[0] = nomeImg[x]
        parametro.has_key(1)
        parametro[1] = nomeRotulo[x]
        parametro.has_key(2)
        parametro[2] = nomeGabarito[x]

        nomeArqResultado = nomeImg[x] + ".csv"
        arqRes = csv.writer(open(nomeArqResultado,"wb"), delimiter=';')
        arqRes.writerow(["nro","PxFundo", "ErrosFundo", "AcertoFundo", "PxObjeto", "ErrosObjeto", "AcertoObjeto", "PxImagem", "PxCinza", "ErrosImagem", "AcertoImagem", "TpFase1", "TpFase2", "TpTotal"])

        for vezes in range(0, nVezes):
            retorno = executa(parametro)
            arqRes.writerow([ str(vezes), str(retorno[0]), str(retorno[1]), str(retorno[2]), str(retorno[3]), str(retorno[4]), str(retorno[5]), str(retorno[6]), str(retorno[9]), str(retorno[8]), str(retorno[7]), str(retorno[10]), str(retorno[11]), str(retorno[12])])

        
sys.setrecursionlimit(1000000)
main()
