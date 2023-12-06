import face_recognition as fr
import cv2
import os 
import numpy
from datetime import datetime


#Crea base de datos 
ruta = 'estudiantes'
mis_imagenes = []
nombres_estudiantes = []
lista_estudiantes = os.listdir(ruta)
#Esto va llenando las listas
for name in lista_estudiantes:
    imagen_actual = cv2.imread(os.path.join(ruta,name))  #Otra opcion (f'{ruta}/{name}') sin os...
    mis_imagenes.append(imagen_actual)
    nombres_estudiantes.append(os.path.splitext(name)[0])

print(nombres_estudiantes)

#Codificar imagenes
def codificar(imagenes):

    #Crear una lista
    lista_codificada = []

    #Pasar todas a rgb
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        #Codificcar la cara
        codificado = fr.face_encodings(imagen)[0]

        #Agregar a la lista
        lista_codificada.append(codificado)

    #Devolver lista codificada
    return lista_codificada

#Registrar los ingresos
def registrar_ingresos(persona):
    f = open('registro.csv', 'r+')#r+ para abrir y para escribir en el archivo
    lista_datos = f.readlines()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])

    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M:%S')
        f.writelines(f'\n{persona},{string_ahora}')


lista_estudiantes_codificada = codificar(mis_imagenes)

#Toma una imagen desde camara elegida
captura = cv2.VideoCapture(0)#Esto toma la foto desde la camara del pc

#Leer la imagen de la camara
sucess, img = captura.read()

if not sucess:
    print('No se ha podido tomar la foto')
else:
    #Reconocer cara en captura
    cara_captura = fr.face_locations(img) #Localiza la cara de la captura

    #Codificar la cara
    cara_captura_codificada = fr.face_encodings(img, cara_captura)

    #Buscar coincidencias entre captura y las imagenes en la DB
    for cara_codificada,cara_ubicacion in zip(cara_captura_codificada, cara_captura):#El zip sirve para hacer varios loops a la vez
        coincidencias = fr.compare_faces(lista_estudiantes_codificada, cara_codificada)
        distancias = fr.face_distance(lista_estudiantes_codificada, cara_codificada)
        
        print(distancias)

        indice_coincidencias = numpy.argmin(distancias)
        
        #Muestra si hay coincidencias
        if distancias[indice_coincidencias] > 0.6:
            print('No coincide con ninguno de nuestros estudiantes en nuestra base de datos')
        
        else:
            #BUscar nombre del empleado
            nombre_estudiante_capturado = nombres_estudiantes[indice_coincidencias]
            #print(nombres_estudiantes[indice_coincidencias])##########
            #
            y1, x2, y2, x1 = cara_ubicacion
            cv2.rectangle(img,
                        (x1, y1),(x2, y2),
                        (0, 255, 0),
                        2)
            cv2.rectangle(img,
                        (x1,y2 - 35),(x2, y2),
                        (0, 255, 0),
                        cv2.FILLED)
            cv2.putText(img,
                        nombre_estudiante_capturado,
                        (x1 + 6,y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1)

            registrar_ingresos(nombre_estudiante_capturado)

            #Mostrarla imagen obtenida
            cv2.imshow('Imagen web', img)

            

            #Mantener la imagen
            cv2.waitKey(0)
            print(f'{nombre_estudiante_capturado} Bienvenido a los angeles te estabamos esperando')



