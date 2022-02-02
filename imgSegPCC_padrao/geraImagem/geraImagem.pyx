from PIL import Image
from datetime import datetime
import scipy.misc
import numpy as np


cpdef gerar(container):

    horaAgora = datetime.now()
    diretorio = "imgResultado/"
    #extraindo parametros
    arrNodes = container[0]
    arrGab = container[1]
    arrImg = container[2]
    imgWidth = container[3]
    imgHeight = container[4]
    imgChannels = container[5]
    nRows = container[6]
    nCols = container[7]
    nClasses = container[8]

    print("Gera imagens resultado...")
    imgSeg = np.zeros((imgHeight, imgWidth, imgChannels), dtype=np.uint8)
    imgFinalRed = np.zeros((nRows, nCols, imgChannels), dtype=np.uint8)
    imgFundo = np.zeros((imgHeight, imgWidth, imgChannels), dtype=np.uint8)

    #gera imagem final
    pNode = np.int32(0)
    for row in range(0,nRows):
        for col in range(0, nCols):
            if (arrNodes[pNode,0] == 0):
                #Fundo da imagem
                imgFinalRed[row, col] = [0,0,0]
            elif (arrNodes[pNode,0] == 1):
                imgFinalRed[row, col] = [255, 255, 255]
            else:
                if(arrNodes[pNode,0] == -1):
                   imgFinalRed[row, col] = [0,255,0]
                else:
                    imgFinalRed[row, col] = [255, 0, 0]
            pNode=pNode+1

    imgFinalRedG = Image.fromarray(imgFinalRed)
    nomeImg = diretorio+"resImgRedFinal_"+str(horaAgora.hour)+str(horaAgora.minute)+str(horaAgora.second)+".png"
    imgFinalRedG.save(nomeImg, "PNG")

    # Retorno a dimensionalidade da imagem
    # interpolacao bicubica da imagem
    imgFinal = scipy.misc.imresize(imgFinalRed, (imgHeight, imgWidth), interp='bicubic')
    arrFinal = np.array(imgFinal, dtype=np.uint8)

    imgbicubic = Image.fromarray(imgFinal)
    nomeImg = diretorio+"resImgbicubic_"+str(horaAgora.hour)+str(horaAgora.minute)+str(horaAgora.second)+".png"
    imgbicubic.save(nomeImg, "PNG")

    contaGray = 0
    contaPxFundo = 0
    contaPxFrente = 0
    contaPxImg = 0
    contaPxObjeto = 0
    contaAcerto = 0
    contaAcertoFundo = 0
    contaAcertoFrente = 0
    contaErroFundo = 0
    contaErroFrente = 0

    # gera imagem resultado
    for j in range(0, imgHeight, 1):
        for i in range(0, imgWidth, 1):
            #print("Gerando pixel x: " + str(j) + " - y: " + str(i))
            contaPxImg = contaPxImg+1
            r, g, b = arrGab[j,i]
            r1,g1,b1 = arrFinal[j,i]
            if (r==128 and g==128 and b==128):
                #pixel cinza faixa de segmentacao nao definida
                contaGray = contaGray+1
                imgSeg[j, i] = arrImg[j, i]
                imgFundo[j, i] = [255, 255, 255]
            elif (r==r1 and g==g1 and b==b1):
                #acertou o pixel
                contaAcerto = contaAcerto+1
                if (r==255 and g==255 and b==255):
                    #Frente objeto a ser segmentado
                    contaPxFrente = contaPxFrente+1
                    contaAcertoFrente = contaAcertoFrente+1
                    imgSeg[j,i] = arrImg[j,i]
                    imgFundo[j,i] = [255,255,255]
                else:
                    #fundo da imagem
                    contaPxFundo = contaPxFundo+1
                    contaAcertoFundo=contaAcertoFundo+1
                    imgSeg[j,i] = [255,255,255]
                    imgFundo[j,i] = arrImg[j,i]
            else:
                #caso errou o pixel
                if (r==255 and g==255 and b==255):
                    #erro objeto a ser segmentado
                    contaPxFrente=contaPxFrente+1
                    contaErroFrente=contaErroFrente+1
                    
                    if (r1==0 and g1==255 and b1==0):
                        imgSeg = [0,255,0]
                    else:
                        imgSeg[j,i] = [255,0,0]
                    
                    imgFundo[j,i] = [255,255,255]

                else:
                    #erro fundo
                    contaPxFundo=contaPxFundo+1
                    contaErroFundo=contaErroFundo+1
                    imgSeg[j,i] = [255,255,255]
                    
                    if(r1==0 and g1==255 and b1==0):
                        imgFundo[j,i] = [0,255,0]
                    else:
                        imgFundo[j,i] = [255,0,0]

    imgSeg = Image.fromarray(imgSeg)
    nomeImg = diretorio+"resImgSegmento_"+str(horaAgora.hour)+str(horaAgora.minute)+str(horaAgora.second)+".png"
    imgSeg.save(nomeImg,"PNG" )
    imgFundo = Image.fromarray(imgFundo)
    nomeImg = diretorio+"resImgSegFundo_"+str(horaAgora.hour)+str(horaAgora.minute)+str(horaAgora.second)+".png"
    imgFundo.save(nomeImg, "PNG")


    resposta={}
    resposta.has_key(0)
    resposta[0] = contaPxFundo
    resposta.has_key(1)
    resposta[1] = contaErroFundo
    resposta.has_key(2)
    resposta[2] = contaAcertoFundo
    resposta.has_key(3)
    resposta[3] = contaPxFrente
    resposta.has_key(4)
    resposta[4] = contaErroFrente
    resposta.has_key(5)
    resposta[5] = contaAcertoFrente
    resposta.has_key(6)
    resposta[6] = contaPxImg
    resposta.has_key(7)
    resposta[7] = (contaAcertoFundo+contaAcertoFrente)
    resposta.has_key(8)
    resposta[8] = (contaErroFrente+contaErroFundo)

    return resposta