import sqlite3

conexion = sqlite3.connect('usuarios_autoincremental.db')

cursor = conexion.cursor()

# Creamos un campo dni como clave primaria
cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY,
                    dni VARCHAR(9) UNIQUE,
                    nombre VARCHAR(100), 
                    edad INTEGER(3),
                    email VARCHAR(100))''')

#usuarios = [('11111111E', 'Hector', 27, 'hector@ejemplo.com'),
#            ('22222222F', 'Mario', 51, 'mario@ejemplo.com'),
#            ('33333333G', 'Mercedes', 38, 'mercedes@ejemplo.com'),
#            ('44444444H', 'Juan', 19, 'juan@ejemplo.com')]

#cursor.executemany("INSERT INTO usuarios VALUES (null, ?,?,?,?)", usuarios)

conexion.commit()
conexion.close()

#####################################################################
################LECTURA MULTIPLE ####################################
conexion = sqlite3.connect('usuarios_autoincremental.db')
cursor = conexion.cursor()

# Recuperamos los registros de la tabla de usuarios
cursor.execute("SELECT * FROM usuarios")

# Recorremos todos los registros con fetchall
# y los volcamos en una lista de usuarios
usuarios = cursor.fetchall()

# Ahora podemos recorrer todos los usuarios
for usuario in usuarios:
    print(usuario)

conexion.close()

#####################################################################
##################CONSULTA CON WHERE ################################
conexion = sqlite3.connect('usuarios_autoincremental.db')
cursor = conexion.cursor()

# Recuperamos un registro de la tabla de usuarios
cursor.execute("SELECT * FROM usuarios WHERE id=7")

usuario = cursor.fetchone()
print(usuario)

conexion.close()

##################CONSULTA CON WHERE ################################
conexion = sqlite3.connect('usuarios_autoincremental.db')
cursor = conexion.cursor()

# Recuperamos un registro de la tabla de usuarios
cursor.execute("SELECT nombre, edad, email FROM usuarios WHERE dni='22222222B'")

usuario = cursor.fetchone()
print(usuario)

conexion.close()

#####################################################################
##################MODIFICACION CON UPDATE ###########################
conexion = sqlite3.connect('usuarios_autoincremental.db')
cursor = conexion.cursor()

# Actualizamos un registro
cursor.execute("UPDATE usuarios SET nombre='Hector Costa R' " \
    "WHERE dni='11111111A'")

# Ahora lo consultamos de nuevo
cursor.execute("SELECT * FROM usuarios WHERE dni='11111111A'")
usuario = cursor.fetchone()
print(usuario)

conexion.commit()
conexion.close()

#####################################################################
#######################BORRADO CON UPDATE ###########################

conexion = sqlite3.connect('usuarios_autoincremental.db')
cursor = conexion.cursor()
# Consultamos los usuarios
for usuario in cursor.execute("SELECT * FROM usuarios"):
    print(usuario)

# Ahora lo borramos
cursor.execute("DELETE FROM usuarios WHERE dni='11111111E'")
print() # Espacio en blanco

# Consultamos de nuevo los usuarios
for usuario in cursor.execute("SELECT * FROM usuarios"):
    print(usuario)

conexion.commit()
conexion.close()