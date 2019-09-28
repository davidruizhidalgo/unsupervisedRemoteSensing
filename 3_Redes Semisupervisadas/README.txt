ESTRUCTURA DE ARCHIVOS
Para ejecutar cualquiera de estos archivos, deben ser copiados en la carpeta de directorio
principal (unsupervisedRemoteSensing/). Igualmente se debe respetar la siguiente estructura de archivos. 

dataSets/
        ||DatosSOM/
                =>...
        ||Indian_pines.mat
        ||Indian_pines_gt.mat
        ||Salinas.mat
        ||Salinas_gt.mat
        ||PaviaU.mat
        ||PaviaU_gt.mat
unsupervisedRemoteSensing/
	||package/
		=>cargarHSI.py
		=>firmasEspectrales.py
		=>MorphologicalProfiles.py
		=>PCA.py
		=>prepararDatos.py
		=>dataLogger.py
	||red1.py
	||red2.py
	||red3.py