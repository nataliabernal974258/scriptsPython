# libreria pandas
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carga del archivo con configuraciones para que funcionen con utf-8
data = pd.read_csv('data.csv', header=0, encoding='unicode_escape')

# Si tuvieran valores nulos
data = data.dropna()

# Asignar a variables valores de columnas especificas del archivo
toc = data.iloc[:, 4].values
animo = data.iloc[:, 5].values
obediencia = data.iloc[:, 6].values
#fecha = data.iloc[:, 8].values

# Instancia de LabelEncoder
LabelEncoder_data = LabelEncoder()

# Borramos columnas que vamos a reemplazar
data.drop(['TDAH_TOP_TC', 'animo', 'obediencia'], axis='columns', inplace=True)

# Escalar volver datos string en numeros
data['TDAH_TOP_TC'] = LabelEncoder_data.fit_transform(toc)
data['animo'] = LabelEncoder_data.fit_transform(animo)
data['obediencia'] = LabelEncoder_data.fit_transform(obediencia)


#castear todas las columnas como int 32
#data = data.astype('int32')
#data['fecha'] = fecha

#castear algunas columnas con el tipo de dato correspondiente
data = data.astype({"id_niño": int, "actividad": int, "categoria_actividad": int, "edad": int, "tiempo_rsp_seg": int, "fecha":'datetime64'})

#filtrar para que las fechas sean seleccionables
data = data[(data['fecha'] > '2020/01/01') & (data['fecha'] <= '2021/04/09')]

#filtrar por edades
data = data[(data['edad'] > 4) & (data['edad'] <= 10)]


#print(data)

#Exporta data a un archivo con pre-procesamiento de la data
data.to_csv('data_clean.csv', index=True, sep='|')  