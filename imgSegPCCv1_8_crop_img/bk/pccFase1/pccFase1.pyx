from collections import defaultdict
from skimage.filters import threshold_otsu
import numpy as np
import math
import scipy.spatial

#cimport numpy as np

def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v

def geraInfluenciaParticula(arrNodes, arrDominio, nroNode, classe, width):
    #calcula posicoes dos nodes na matriz
    pos = []
    posicao = 1
    #1 faixa 
    pos.append(nroNode-(width*posicao)-1)
    pos.append(nroNode-(width*posicao))
    pos.append(nroNode-(width*posicao)+1)
    pos.append(nroNode-1)
    pos.append(nroNode+1)
    pos.append(nroNode+(width*posicao)-1)
    pos.append(nroNode+(width*posicao))
    pos.append(nroNode+(width*posicao)+1)

    pos1 = []
    posicao = 2
    #2 faixa
    pos1.append(nroNode-(width*posicao)-2)
    pos1.append(nroNode-(width*posicao)-1)
    pos1.append(nroNode-(width*posicao))
    pos1.append(nroNode-(width*posicao)+1)
    pos1.append(nroNode-(width*posicao)+2)

    pos1.append(nroNode-2)
    pos1.append(nroNode-1)
    pos1.append(nroNode+1)
    pos1.append(nroNode+2)

    pos1.append(nroNode+(width*posicao)-2)
    pos1.append(nroNode+(width*posicao)-1)
    pos1.append(nroNode+(width*posicao))
    pos1.append(nroNode+(width*posicao)+1)
    pos1.append(nroNode+(width*posicao)+2)

    for x in range(0,8):
        if(pos[x]>=0 and pos[x]<=len(arrNodes)):
            #se nao for rotulado pelo usuario
            if(arrNodes[pos[x],1]==0):
                #atualiza dominio
                arrNodes, arrDominio = atualizaDominio(arrNodes,arrDominio,pos[x],classe,0.2)

    for x in range(0,14):
        if(pos1[x]>=0 and pos1[x]<=len(arrNodes)):
            #se nao for rotulado pelo usuario
            if(arrNodes[pos1[x],1]==0):
                #atualzia dominio
                arrNodes, arrDominio = atualizaDominio(arrNodes,arrDominio,pos1[x],classe,0.1)

    return (arrNodes,arrDominio)

def atualizaDominio(arrNodes, arrDominio, nroNode, classe, peso):
    if (nroNode>=0 and nroNode<=len(arrNodes)):
        classe0 = arrDominio[nroNode,0]
        classe1 = arrDominio[nroNode,1]

        if(classe==0):
            classe0=classe0+peso
            classe1=classe1-peso
        else:
            classe0 = classe0 - peso
            classe1 = classe1 + peso

        if(classe0>1):
            classe0 = 1
            classe1 = 0
        elif(classe1>1):
            classe0 = 0
            classe1 = 1

        arrDominio[nroNode,0] = classe0
        arrDominio[nroNode,1] = classe1

        if(classe0>classe1):
            arrNodes[nroNode,0] = 0
        elif(classe1>classe0):
            arrNodes[nroNode,0] = 1
        else:
            arrNodes[nroNode,0] = -1

    return (arrNodes, arrDominio)

cpdef pccFase1(container):

    img = container[0]
    imgRotulo = container[1]
    nClasses = container[2]
    k = container[3]
    verificaLigacao = container[4]
    featLength = container[5]
    nRows = container[6]
    nCols = container[7]
    weghts = container[8]
    arrGray = container[9]

    size = nRows*nCols

    #Fase1 do algoritmo de competicao e cooperacao de particulas
    force=range(nClasses)
    #valor padrao para nivel de dominios em nos
    for n in range(0,nClasses):
        force[n]=np.double((1.00/float(nClasses)))
    
    #tamanho da matriz da imagem
    #dim = img.size
    #nRows = dim[0]
    #nCols = dim[1]
    
    #gera binarizacao otsu
    print("Gera imagem binarizada Otsu...")
    thOtsu = threshold_otsu(arrGray)
    maskOtsu = (arrGray<thOtsu)
    arrGray[~maskOtsu] = 255
    arrGray[maskOtsu] = 0

    print("Gera Vertices e extrai caracteristicas dos pixels...")
    #declara matriz de vertices do grafo
    #[dominio atual, no rotulado]
    arrNodes = np.ndarray(shape=(size,2),dtype=np.int32)
    #arrRotulado = np.empty([size],dtype=np.int32)
    #arrClasse = np.empty([size],dtype=np.int32)
    #[class 0, class 1, class 2, ... , class n]
    arrDominio = np.ndarray(shape=(size,nClasses),dtype=np.double)

    #declara matriz de caracteristicas
    #[x, y, r, g, b, v, exr, exg, exb]
    vOtsu = np.empty([size], dtype=np.double)
    vLin = np.empty([size], dtype=np.double)
    vCol = np.empty([size], dtype=np.double)
    vRed = np.empty([size], dtype=np.double)
    vGreen = np.empty([size], dtype=np.double)
    vBlue = np.empty([size], dtype=np.double)
    vValue = np.empty([size], dtype=np.double)
    vExR = np.empty([size], dtype=np.double)
    vExG = np.empty([size], dtype=np.double)
    vExB = np.empty([size], dtype=np.double)
    
    #variaveis de controle
    cdef int acumParticula = 0
    cdef int contaNo = 0

    print("x = "+str(nRows))
    print("y = "+str(nCols))
    
    #laco para extracao de caracteristicas e geracao dos vertices do modelo
    for x in range(nRows):
        for y in range(nCols):
            #extrai rgb
            r, g, b = img[x, y]

            vLin[contaNo] = x #x
            vCol[contaNo] = y #y
            vRed[contaNo] = r #r
            vGreen[contaNo] = g #g
            vBlue[contaNo] = b #b

            #hsv
            h, s, v = rgb2hsv(r,g,b)
            vValue[contaNo] = v #v

            # ExR, ExG, ExB
            vExR[contaNo] = (2 * int(r) - (int(g) + int(b)))  # ExR
            vExG[contaNo] = (2 * int(g) - (int(r) + int(b)))  # ExG
            vExB[contaNo] = (2 * int(b) - (int(r) + int(g)))  # ExB

            #extrai a escala de cinza da imagem gerada OTSU
            vOtsu[contaNo] = arrGray[x,y]

            #determina dominios para o vertice
            for cl in range(nClasses):
                arrDominio[contaNo,cl] = force[cl]

            #gera o vertice na matriz
            arrNodes[contaNo,0] = -1 #sem dominio
            arrNodes[contaNo,1] = 0 #nao rotulado pelo usuario

            r1,g1,b1 = imgRotulo[x,y]
            if ((r1==255 and g1==0 and b1==0) or (r1==0 and g1==255 and b1==0)):
                acumParticula=acumParticula+1

            contaNo=contaNo+1

    print("Gera as particulas -->" + str(acumParticula))
    #gera array de particulas
    #[pos atual, forca, no casa, classe]
    arrParticle = np.ndarray(shape=(acumParticula,4),dtype=np.double)
    #laco geracao de particulas
    cdef int contaParticula = 0
    contaNo = 0
    for x in range(nRows):
        for y in range(nCols):
            r1,g1,b1 = imgRotulo[x,y]
            if (r1==255 and g1==0 and b1==0):
                #fundo
                arrParticle[contaParticula,0] = contaNo
                arrParticle[contaParticula,1] = 1
                arrParticle[contaParticula,2] = contaNo
                arrParticle[contaParticula,3] = 0
                contaParticula=contaParticula+1

                #dominios
                arrDominio[contaNo,0] = 1 #forca 1 para classe 0
                arrDominio[contaNo,1] = 0 #forca 0 para classe 1
                #vertices
                arrNodes[contaNo,0] = 0 #classe dominante 0
                arrNodes[contaNo,1] = 1 #no rotulado pelo usuario

                arrNodes, arrDominio = geraInfluenciaParticula(arrNodes, arrDominio, contaNo, 0, nCols)

            elif (r1==0 and g1==255 and b1==0):
                #objeto(frente)
                arrParticle[contaParticula,0] = contaNo
                arrParticle[contaParticula,1] = 1
                arrParticle[contaParticula,2] = contaNo
                arrParticle[contaParticula,3] = 1
                contaParticula=contaParticula+1

                #dominios
                arrDominio[contaNo,0] = 0 #forca 0 para a classe 0
                arrDominio[contaNo,1] = 1 #forca 1 para a classe 1
                #vertices
                arrNodes[contaNo,0] = 1 #classe dominio 1
                arrNodes[contaNo,1] = 1 #no rotulado pelo usuario

                arrNodes, arrDominio = geraInfluenciaParticula(arrNodes, arrDominio, contaNo, 1, nCols)

            contaNo=contaNo+1

    print("Normaliza caracteristicas...")
    #normaliza as caracteristicas media = 0 e desvio padrao = 1
    
    if(weghts):
       vNorm_Lin = featLength[0] * (vLin-np.mean(vLin))/np.std(vLin)
       vNorm_Col = featLength[1] * (vCol-np.mean(vCol))/np.std(vCol)
       vNorm_Red = featLength[2] * (vRed-np.mean(vRed))/np.std(vRed)
       vNorm_Green = featLength[3] * (vGreen-np.mean(vGreen))/np.std(vGreen)
       vNorm_Blue = featLength[4] * (vBlue-np.mean(vBlue))/np.std(vBlue)
       vNorm_Value = featLength[5] * (vValue-np.mean(vValue))/np.std(vValue)
       vNorm_ExR = featLength[6] * (vExR-np.mean(vExR))/np.std(vExR)
       vNorm_ExG = featLength[7] * (vExG-np.mean(vExG))/np.std(vExG)
       vNorm_ExB = featLength[8] * (vExB-np.mean(vExB))/np.std(vExB)
    else:
        vNorm_Otsu = (vOtsu-np.mean(vOtsu)/np.std(vOtsu))
        vNorm_Lin = (vLin-np.mean(vLin))/np.std(vLin)
        vNorm_Col = (vCol-np.mean(vCol))/np.std(vCol)
        vNorm_Red = (vRed-np.mean(vRed))/np.std(vRed)
        vNorm_Green = (vGreen-np.mean(vGreen))/np.std(vGreen)
        vNorm_Blue = (vBlue-np.mean(vBlue))/np.std(vBlue)
        vNorm_Value = (vValue-np.mean(vValue))/np.std(vValue)
        vNorm_ExR = (vExR-np.mean(vExR))/np.std(vExR)
        vNorm_ExG = (vExG-np.mean(vExG))/np.std(vExG)
        vNorm_ExB = (vExB-np.mean(vExB))/np.std(vExB)

    #gera a matriz de caracteristicas normalizadas de cada vertice
    arrFeatures = zip(vNorm_Otsu,vNorm_Lin,vNorm_Col,vNorm_Red,vNorm_Green,vNorm_Blue,vNorm_Value,vNorm_ExR,vNorm_ExG,vNorm_ExB)
    
    print("Gera k vizinhos proximos.... ")
    #matriz de vizinhos do grafo (arestas)
    #arrNeig = np.ndarray(shape=(size,k),dtype=np.int32)
    arrNeig = np.full((size,size), 0)

    #Gerando vizinhos proximos
    #gera a matriz de busca de vizinhos por kdtree
#    tree = scipy.spatial.cKDTree(arrEuclidean, leafsize=40)
    tree = scipy.spatial.cKDTree(arrFeatures, leafsize=40)
    cdef int classeVertice = -1
    cdef int classeVizinho = -1
    for no in range(size):
        print("Arestas --> Vertice :" + str(no) + "/" + str(size))
        print("A")
        classeVertice = arrNodes[no,0]
        #gera dicionario de vizinhos para o vertice
        #gera vizinhos proximos pelo peso
        d, i = tree.query(arrFeatures[no], k=(k+1))

        #adiciona 8 vizinhos proximos (relacao fisica de proximidade)
        #realiza a ligacao do vizinho fisico ao vertice unidirecional
        if ( (no-1)>=0 and (no-1)<=size-1):
            #Posicao -1 coluna na mesma linha
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no-1),0]==-1 or classeVertice==arrNodes[(no-1),0]):
                    arrNeig[(no-1), no] = 1
                    arrNeig[no, (no-1)] = 1
            else:
                arrNeig[(no-1), no] = 1
                arrNeig[no, (no-1)] = 1

        if ( (no+1)>=0 and (no+1)<=size-1):
            #posicao +1 coluna na mesma linha
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no+1),0]==-1 or classeVertice==arrNodes[(no+1),0]):
                    arrNeig[(no+1), no] = 1
                    arrNeig[no, (no+1)] = 1
            else:
                arrNeig[(no+1), no] = 1
                arrNeig[no, (no+1)] = 1

        if ( (no-nCols)>=0 and (no-nCols)<=size-1):
            #posicao mesma coluna -1 linha
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no-nCols),0]==-1 or classeVertice==arrNodes[(no-nCols),0]):
                    arrNeig[(no-nCols), no] = 1
                    arrNeig[no, (no-nCols)] = 1
            else:
                arrNeig[(no-nCols), no] = 1
                arrNeig[no, (no-nCols)] = 1

        if ( (no+nCols)>=0 and (no+nCols)<=size-1):
            #posicao mesma coluna + 1 linha
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no+nCols),0]==-1 or classeVertice==arrNodes[(no+nCols),0]):
                    arrNeig[(no+nCols), no] = 1
                    arrNeig[no, (no+nCols)] = 1
            else:
                arrNeig[(no+nCols), no] = 1
                arrNeig[no, (no+nCols)] = 1

        if ( (no-nCols-1)>=0 and (no-nCols-1)<=size-1):
            #posicao -1 linha e -1 coluna
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no-nCols-1),0]==-1 or classeVertice==arrNodes[(no-nCols-1),0]):
                    arrNeig[(no-nCols-1), no] = 1
                    arrNeig[no, (no-nCols-1)] = 1
            else:
                arrNeig[(no-nCols-1), no] = 1
                arrNeig[no, (no-nCols-1)] = 1

        if ( (no-nCols+1)>=0 and (no-nCols+1)<=size-1):
            #posicao -1 linha e +1 coluna
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no-nCols+1),0]==-1 or classeVertice==arrNodes[(no-nCols+1),0]):
                    arrNeig[(no-nCols+1), no] = 1
                    arrNeig[no, (no-nCols+1)] = 1
            else:
                arrNeig[(no-nCols+1), no] = 1
                arrNeig[no, (no-nCols+1)] = 1

        if ( (no+nCols-1)>=0 and (no+nCols-1)<=size-1):
            #posicao +1 linha e -1 coluna
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no+nCols-1),0]==-1 or classeVertice==arrNodes[(no+nCols-1),0]):
                    arrNeig[(no+nCols-1), no] = 1
                    arrNeig[no, (no+nCols-1)] = 1
            else:
                arrNeig[(no+nCols-1), no] = 1
                arrNeig[no, (no+nCols-1)] = 1

        if ( (no+nCols+1)>=0 and (no+nCols+1)<=size-1):
            #posicao +1 linha e -1 coluna
            if (verificaLigacao):
                if (classeVertice==-1 or arrNodes[(no+nCols+1),0]==-1 or classeVertice==arrNodes[(no+nCols+1),0]):
                    arrNeig[(no+nCols+1), no] = 1
                    arrNeig[no, (no+nCols+1)] = 1
            else:
                arrNeig[(no+nCols+1), no] = 1
                arrNeig[no, (no+nCols+1)] = 1

        print("B")
        for nViz in range(i.size):
            if (i[nViz]!=no):
                vizinho = i[nViz]
                classeVizinho = arrNodes[vizinho,0]
                #verifica se liga vizinho de classes diferentes
                if (verificaLigacao):
                    if (classeVertice==-1 or classeVizinho==-1 or classeVertice==classeVizinho):
                        #ativa ligacao bi-direcional entre os vertices
                        arrNeig[no,vizinho] = 1
                        arrNeig[vizinho,no] = 1
                else:
                    #ativa ligacao bi-direcional entre os vertices
                    arrNeig[no,vizinho] = 1
                    arrNeig[vizinho,no] = 1

    print("Gera matriz de distancias... ")
    #matriz de distancias da particula
    distMax = size-1
    arrDistancia = np.full((acumParticula,size),distMax)
    for nPart in range(0, acumParticula):
        noCasa = np.int32(arrParticle[nPart,0])
        arrDistancia[nPart,noCasa] = 0

    retorno={}
    retorno.has_key(0)
    retorno[0] = arrNodes
    print("Vertices -->"+str(len(arrNodes)))
    retorno.has_key(1)
    retorno[1] = arrDominio
    print("Dominios -->"+str(len(arrDominio)))
    retorno.has_key(2)
    retorno[2] = arrNeig
    print("Vizinhos -->"+str(len(arrNeig)))
    retorno.has_key(3)
    retorno[3] = arrParticle
    print("Particulas -->"+str(len(arrParticle)))
    retorno.has_key(4)
    retorno[4] = arrDistancia
    print("Distancia -->"+str(len(arrDistancia)))
    return (retorno)