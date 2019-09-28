import sqlite3

conexion = sqlite3.connect('productos.db')

cursor = conexion.cursor()

# Las cláusulas not null indican que no puede ser campos vacíos
cursor.execute('''CREATE TABLE IF NOT EXISTS productos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    nombre VARCHAR(100) NOT NULL, 
                    marca VARCHAR(50) NOT NULL, 
                    precio FLOAT NOT NULL)''')

conexion.close()

#########################################################
####### INSERCION MULTIPLE AUTOINCREMENTAL ##############
conexion = sqlite3.connect('productos.db')

cursor = conexion.cursor()

productos = [('Teclado', 'Logitech', 19.95),
            ('Pantalla 19"','LG', 89.95),
            ('Altavoces 2.1','LG', 24.95),]

cursor.executemany("INSERT INTO productos VALUES (null,?,?,?)", productos)

conexion.commit()
conexion.close()


#####################################################################
################LECTURA MULTIPLE ####################################
conexion = sqlite3.connect('productos.db')
cursor = conexion.cursor()

# Recuperamos los registros de la tabla de usuarios
cursor.execute("SELECT * FROM productos")

# Recorremos todos los registros con fetchall
# y los volcamos en una lista de usuarios
productos = cursor.fetchall()

# Ahora podemos recorrer todos los usuarios
for producto in productos:
    print(producto)

conexion.close()

