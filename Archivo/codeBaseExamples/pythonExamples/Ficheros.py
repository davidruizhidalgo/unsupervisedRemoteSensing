from io import open

texto = "Una línea con texto\nOtra línea con texto"

# Ruta donde crearemos el fichero, w indica escritura (puntero al principio)
fichero = open('fichero.txt','w')  

# Escribimos el texto
fichero.write(texto) 

# Cerramos el fichero
fichero.close() 

# Ruta donde leeremos el fichero, r indica lectura (por defecto ya es r)
fichero = open('fichero.txt','r')  

# Lectura completa
#texto2 = fichero.read() 
texto3 = fichero.readlines()
# Cerramos el fichero
fichero.close()  


fichero = open('fichero.txt','a+')  
fichero.write('\nOtra línea más abajo del todo')
fichero.close()

fichero = open('fichero.txt','r')  
fichero.seek(0)
texto4=fichero.read()
fichero.close()
print(texto3)
print(texto4)



# Creamos un fichero de prueba con 4 líneas
fichero = open('fichero2.txt','w')
texto = "Línea 1\nLínea 2\nLínea 3\nLínea 4"
fichero.write(texto)
fichero.close()

# Lo abrimos en lectura con escritura y escribimos algo
fichero = open('fichero2.txt','r+')
fichero.write("0123456")

# Volvemos a ponter el puntero al inicio y leemos hasta el final
fichero.seek(0)
fichero.read()
fichero.close()

fichero = open('fichero2.txt','r+')
texto = fichero.readlines()

# Modificamos la línea que queramos a partir del índice
texto[2] = "Esta es la línea 3 modificada\n"

# Volvemos a ponter el puntero al inicio y reescribimos
fichero.seek(0)
fichero.writelines(texto)
fichero.close()

# Leemos el fichero de nuevo
with open("fichero2.txt", "r") as fichero:
    print(fichero.read())