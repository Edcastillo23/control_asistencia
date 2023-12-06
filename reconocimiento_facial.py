from cv2 import cv2
import face_recognition as fr

#Subir imagen
foto_control = fr.load_image_file('liliana_1.jpg')
foto_prueba = fr.load_image_file('liliana_3.jpg')

#Transforma color de la imagen a formato RGB
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#Localizar cara
lugar_cara_A = fr.face_locations(foto_control)[0]#En consola esto arroja los puntos donde se encuantra la cara
#Para ver la localizacion activar este print
print(lugar_cara_A)
cara_codificada_A = fr.face_encodings(foto_control)[0] # El sistema identifica la cara

#Repetimos el proceso anterior
lugar_cara_B = fr.face_locations(foto_prueba)[0]#En consola esto arroja los puntos donde se encuantra la cara
#Para ver la localizacion activar este print
#print(lugar_cara_B)
cara_codificada_B = fr.face_encodings(foto_prueba)[0] # El sistema identifica la cara

#Muestra el rectangulo de cada cara
cv2.rectangle(foto_control,
            (lugar_cara_A[3],lugar_cara_A[0]),
            (lugar_cara_A[1],lugar_cara_A[2]),
            (0, 255,0,),#Color en que se dibuja el rectagulo
            2)# El grosor del rectangulo


cv2.rectangle(foto_prueba,
            (lugar_cara_B[3],lugar_cara_B[0]),
            (lugar_cara_B[1],lugar_cara_B[2]),
            (0, 255,0,),#Color en que se dibuja el rectagulo
            2)# El grosor del rectangulo



#Realizar comparacion
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B)#Se usa[]porque el primer parámetro es una lista
#print(resultado)#para que lo reconozca como True debe ser menor a 0,6 

#medida de distancia Esta es usada para comparar las caras
distancia = fr.face_distance([cara_codificada_A],cara_codificada_B)
#print(distancia)

#Mostrar imagenes
cv2.putText(foto_prueba,#foto
            f'{resultado}{distancia.round(2)}',#Texto a mostrar
            (50,50),#Ubicacion
            cv2.FONT_HERSHEY_COMPLEX,#Fuente
            1,#Escala
            (0, 255, 0), #Color
            2)# Grosor

#Para mostrar la imagen
cv2.imshow('Liliana 1',foto_control)#Este nombre se muestra en la ventana
cv2.imshow('liliana 2',foto_prueba)

#Esta linea no es necesaria pero queremos mantener abierta las imagenes
cv2.waitKey(0)


