# Importamos el módulo
import sqlite3

conexion = sqlite3.connect('usuarios.db')

cursor = conexion.cursor()

# Creamos un campo dni como clave primaria
cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                    dni VARCHAR(9) PRIMARY KEY,
                    nombre VARCHAR(100), 
                    edad INTEGER,
                    email VARCHAR(100))''')

usuarios = [('11111111E', 'Hector', 27, 'hector@ejemplo.com'),
            ('22222222F', 'Mario', 51, 'mario@ejemplo.com'),
            ('33333333G', 'Mercedes', 38, 'mercedes@ejemplo.com'),
            ('44444444H', 'Juan', 19, 'juan@ejemplo.com')]

cursor.executemany("INSERT INTO usuarios VALUES (?,?,?,?)", usuarios)

conexion.commit()
conexion.close()