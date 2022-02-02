from collections import defaultdict
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
    

    print("Gera Vertices e extrai caracteristicas dos pixels...")
    #declara matriz de vertices do grafo
    #[dominio atual, no rotulado]
    arrNodes = np.ndarray(shape=(size,2),dtype=np.int32)
    #[class 0, class 1, class 2, ... , class n]
    arrDominio = np.ndarray(shape=(size,nClasses),dtype=np.double)

    #declara matriz de caracteristicas
    #[x, y, r, g, b, v, exr, exg, exb]
    vLin = np.empty([size], dtype=np.double)
    vCol = np.empty([size], dtype=np.double)
    vRed = np.empty([size], dtype=np.double)
    vGreen = np.empty([size], dtype=np.double)
    vBlue = np.empty([size], dtype=np.double)
    vHue = np.empty([size], dtype=np.double)
    vSat = np.empty([size], dtype=np.double)
    vValue = np.empty([size], dtype=np.double)
    vMedRed = np.empty([size], dtype=np.double)
    vDvPRed = np.empty([size], dtype=np.double)
    vMedGreen = np.empty([size], dtype=np.double)
    vDvPGreen = np.empty([size], dtype=np.double)
    vMedBlue = np.empty([size], dtype=np.double)
    vDvPBlue = np.empty([size], dtype=np.double)
    vMedHue = np.empty([size], dtype=np.double)
    vDvPHue = np.empty([size], dtype=np.double)
    vMedSat = np.empty([size], dtype=np.double)
    vDvPSat = np.empty([size], dtype=np.double)
    vMedValue = np.empty([size], dtype=np.double)
    vDvPValue = np.empty([size], dtype=np.double)
    vExR = np.empty([size], dtype=np.double)
    vExG = np.empty([size], dtype=np.double)
    vExB = np.empty([size], dtype=np.double)
    
    #variaveis de controle
    cdef int acumParticula = 0
    cdef int contaNo = 0

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
            vHue[contaNo] = h
            vSat[contaNo] = s
            vValue[contaNo] = v #v


            #media e Desvio padrao do RGB e HSV
            contar=0
            somaR=[]
            somaB=[]
            somaG=[]
            somaH=[]
            somaS=[]
            somaV=[]

            for xMedRGB in range(-1,2):
                for yMedRGB in range(-1,2):
                    if (x+xMedRGB>=0 and y+yMedRGB>=0 and x+xMedRGB<nRows and y+yMedRGB<nCols):
                        rm, gm, bm = img[x+xMedRGB, y+yMedRGB]
                        hm, sm, vm = rgb2hsv(rm,gm,bm)
                        somaR.append(rm)
                        somaG.append(gm)
                        somaB.append(bm)
                        somaH.append(hm)
                        somaS.append(sm)
                        somaV.append(vm)
                        contar=contar+1

            var_R = np.var(somaR)
            mean_R = np.average(somaR)
            stdDev_R = np.sqrt(var_R)
            vMedRed[contaNo] = mean_R
            vDvPRed[contaNo] = stdDev_R

            var_G = np.var(somaG)
            mean_G = np.average(somaG)
            stdDev_G = np.sqrt(var_G)
            vMedGreen[contaNo] = mean_G
            vDvPGreen[contaNo] = stdDev_G

            var_B = np.var(somaB)
            mean_B = np.average(somaB)
            stdDev_B = np.sqrt(var_B)
            vMedBlue[contaNo] = mean_B
            vDvPBlue[contaNo] = stdDev_B

            var_H = np.var(somaH)
            mean_H = np.average(somaH)
            stdDev_H = np.sqrt(var_H)
            vMedHue[contaNo] = mean_H
            vDvPHue[contaNo] = stdDev_H

            var_S = np.var(somaS)
            mean_S = np.average(somaS)
            stdDev_S = np.sqrt(var_S)
            vMedSat[contaNo] = mean_S
            vDvPSat[contaNo] = stdDev_S

            var_V = np.var(somaV)
            mean_V = np.average(somaV)
            stdDev_V = np.sqrt(var_V)
            vMedValue[contaNo] = mean_V
            vDvPValue[contaNo] = stdDev_V

            # ExR, ExG, ExB
            vExR[contaNo] = (2 * int(r) - (int(g) + int(b)))  # ExR
            vExG[contaNo] = (2 * int(g) - (int(r) + int(b)))  # ExG
            vExB[contaNo] = (2 * int(b) - (int(r) + int(g)))  # ExB

            #determina dominios para o vertice
            for cl in range(nClasses):
                arrDominio[contaNo,cl] = force[cl]
                
            #gera o vertice na matriz
            arrNodes[contaNo,0] = -1 #sem dominio
            arrNodes[contaNo,1] = 0 #nao rotulado pelo usuario

            r1,g1,b1,a = imgRotulo[x,y]
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
            r1,g1,b1, a = imgRotulo[x,y]
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
        vNorm_Lin = (vLin-np.mean(vLin))/np.std(vLin)
        vNorm_Col = (vCol-np.mean(vCol))/np.std(vCol)
        vNorm_Red = (vRed-np.mean(vRed))/np.std(vRed)
        vNorm_Green = (vGreen-np.mean(vGreen))/np.std(vGreen)
        vNorm_Blue = (vBlue-np.mean(vBlue))/np.std(vBlue)
        vNorm_Hue = (vHue-np.mean(vHue))/np.std(vHue)
        vNorm_Sat = (vSat-np.mean(vSat))/np.std(vSat)
        vNorm_Value = (vValue-np.mean(vValue))/np.std(vValue)
        vNorm_MedR = (vMedRed-np.mean(vMedRed))/np.std(vMedRed)
        vNorm_DvPR = (vDvPRed-np.mean(vDvPRed))/np.std(vDvPRed)
        vNorm_MedG = (vMedGreen-np.mean(vMedGreen))/np.std(vMedGreen)
        vNorm_DvPG = (vDvPGreen-np.mean(vDvPGreen))/np.std(vDvPGreen)
        vNorm_MedB = (vMedBlue-np.mean(vMedBlue))/np.std(vMedBlue)
        vNorm_DvPB = (vDvPBlue-np.mean(vDvPBlue))/np.std(vDvPBlue)
        vNorm_MedH = (vMedHue-np.mean(vMedHue))/np.std(vMedHue)
        vNorm_DvPH = (vDvPBlue-np.mean(vDvPHue))/np.std(vDvPHue)
        vNorm_MedS = (vMedSat-np.mean(vMedSat))/np.std(vMedSat)
        vNorm_DvPS = (vDvPSat-np.mean(vDvPSat))/np.std(vDvPSat)
        vNorm_MedV = (vMedValue-np.mean(vMedValue))/np.std(vMedValue)
        vNorm_DvPV = (vDvPValue-np.mean(vDvPValue))/np.std(vDvPValue)        
        vNorm_ExR = (vExR-np.mean(vExR))/np.std(vExR)
        vNorm_ExG = (vExG-np.mean(vExG))/np.std(vExG)
        vNorm_ExB = (vExB-np.mean(vExB))/np.std(vExB)

    #gera a matriz de caracteristicas normalizadas de cada vertice
    arrFeatures = zip(vNorm_Lin,vNorm_Col,vNorm_Red,vNorm_Green,vNorm_Blue,vNorm_Hue,vNorm_Sat,vNorm_Value,vNorm_MedR,vNorm_DvPR,vNorm_MedG,vNorm_DvPG,vNorm_MedB,vNorm_DvPB,vNorm_MedH,vNorm_DvPH,vNorm_MedS,vNorm_DvPS,vNorm_MedV,vNorm_DvPV,vNorm_ExR,vNorm_ExG,vNorm_ExB)
    
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
        #print("Arestas --> Vertice :" + str(no) + "/" + str(size))
        classeVertice = arrNodes[no,0]
        #gera dicionario de vizinhos para o vertice
        #gera vizinhos proximos pelo peso
        d, i = tree.query(arrFeatures[no], k=(k+1))
        
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