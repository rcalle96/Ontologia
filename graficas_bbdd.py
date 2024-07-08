# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:54:36 2024

@author: raul.calle
"""

# Manipulacion de carpetas
# ==============================================================================
import os
import shutil

# Tratamiento de df
# ==============================================================================
import pandas as pd
import numpy as np

# Importacion df
# ==============================================================================
import tkinter as tk
from tkinter import filedialog as fd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import geopandas as gpd

import json


def importar_archivo():
    try:
        root = tk.Tk()
        root.withdraw()
        files = fd.askopenfilename()
        root.destroy()
        if files:
            print(f'Archivo importado con éxito\nArchivo: {files}')
            return files
        else:
            print('No se han podido importar el archivo')
            return None
    except Exception as e:
        print(f'Error al importar archivos: {e}')
        return None


def crear_mapa_cloropletas(datos, columna_color, titulo='', etiqueta_color='', leyenda=True):
    """
    Crea un mapa de cloropletas a partir de un DataFrame geoespacial.

    Argumentos:
    datos : GeoDataFrame
        El DataFrame geoespacial que contiene los datos para trazar en el mapa de cloropletas.
    columna_color : str
        El nombre de la columna en el DataFrame que se utilizará para colorear el mapa.
    titulo : str, opcional
        El título del mapa. Por defecto es una cadena vacía.
    etiqueta_color : str, opcional
        La etiqueta que se mostrará en la barra de color. Por defecto es una cadena vacía.
    leyenda : bool, opcional
        Indica si se debe mostrar la leyenda del mapa. Por defecto es True.

    Retorna:
    None
    """
    # Crear el gráfico
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Trazar el mapa de cloropletas
    datos.plot(column=columna_color, cmap='bwr', linewidth=0.8, ax=ax, edgecolor='0.8', legend=leyenda)

    # Añadir título y etiquetas
    ax.set_title(titulo)
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')

    # Añadir barra de color
    if leyenda:
        sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=datos[columna_color].min(), vmax=datos[columna_color].max()))
        sm._A = []
        cbar = fig.colorbar(sm)
        cbar.set_label(etiqueta_color)

    # Mostrar el gráfico
    plt.show()


# =============================================================================
# Configuración matplotlib
# =============================================================================
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['figure.dpi'] = "150"
plt.rcParams['savefig.bbox'] = "tight"
style.use('default') or plt.style.use('default')

# =============================================================================
# IMPORTAR EL ARCHIVO
# =============================================================================
papers_csv = "C://Users//raul.calle//Documents//RAUL//Base de Datos//paper_references (1).csv"

# =============================================================================
# LEER df
# =============================================================================
column_names = ['ID', 'valid', 'no_ref', 'new_ref', 'filename', 'DOI', 'title',
                'year', 'journal', 'pages', 'issue', 'abstract', 'authors',
                'volume', 'corresponding_author', 'country', 'country_name']
df_refs = pd.read_csv(papers_csv, names=column_names)


# =============================================================================
# COMPLETAR EL DF
# =============================================================================
func_separacion = lambda x: x.replace('"', '').split(',')
df_refs.loc[:, ['country', 'country_name']] = df_refs.loc[:, ['country', 'country_name']].map(func_separacion, na_action='ignore')

# =============================================================================
# GRAFICA PUBLICACIONES POR AÑO
# =============================================================================
years = df_refs['year'].sort_values().unique()
no_refs_year = df_refs['year'].value_counts().sort_values().cumsum()

refs_year = pd.DataFrame({'year': years, 'no_refs': no_refs_year})

g_year = sns.catplot(data=refs_year, x='year', y='no_refs',
                     kind='bar', height=5, aspect=1.5, palette='turbo')
g_year.tick_params('x', labelsize=10, labelrotation=90)
g_year.set_axis_labels(x_var='Year', y_var='Publications')
g_year.despine(top=False, right=False)
g_year.set(title='Publications per year')

# =============================================================================
# GRAFICA PUBLICACIONES POR PAIS
# =============================================================================
ruta_mapamundi_json = "C://Users//raul.calle//Documents//RAUL//Base de Datos//gistfile1.js"
with open(ruta_mapamundi_json, 'r') as file:
    world_map = json.load(file)
gdf = gpd.GeoDataFrame.from_features(world_map)


ids = [d['id'] for d in world_map['features']]
country_names = [d['properties']['name'] for d in world_map['features']]
country_ids = dict(zip(ids, country_names))


# Limpiar columnas de paises y aislar el primer elemento
func_seleccion = lambda x: x[0]
func_nombre = lambda x: country_ids[x]
df_refs_country = df_refs.copy()
df_refs_country['country'] = df_refs_country['country'].map(func_seleccion, na_action='ignore')
df_refs_country['pais'] = df_refs_country['country'].map(func_nombre, na_action='ignore')

country = df_refs_country['country'].value_counts().index
no_refs_country = df_refs_country['country'].value_counts()

country = []
no_refs_country = []
country_name = []
for c, refs in df_refs_country.groupby('country'):
    country.append(c)
    no_refs_country.append(refs.DOI.value_counts().sum())
    country_name.append(refs.pais.unique()[0])

refs_country = pd.DataFrame({'id': country,
                             'name': country_name,
                             'no_refs': no_refs_country})

gdf_refs = gdf.merge(refs_country, how='outer', on='name')

ax = gdf_refs.plot(column='no_refs', cmap='OrRd', legend=True,
                   legend_kwds={"label": "Publicaciones",
                                "orientation": "horizontal"},
                   missing_kwds={"color": "lightgrey",
                                 "edgecolor": "red",
                                 "hatch": "///",
                                 "label": "Missing values"})
gdf.boundary.plot(ax=ax, linewidth=0.5, color='k')
ax.set_axis_off()
ax.set_title('Publications per country')
# =============================================================================
# GRAFICA PUBLICACIONES POR REVISTA
# =============================================================================
journals = df_refs['journal'].sort_values().unique()
no_refs_journal = df_refs['journal'].sort_values().value_counts()

refs_journal = pd.DataFrame({'journal': journals, 'no_refs': no_refs_journal})
refs_journal.sort_values(by='no_refs', ascending=False, inplace=True)

# Criba de las referencias
criba = refs_journal['no_refs'] >= 10
refs_journal_cribado = refs_journal.loc[criba, :]

g_journal = sns.catplot(data=refs_journal_cribado, y='journal', x='no_refs',
                        kind='bar', height=5, aspect=0.5)
g_journal.tick_params('y', labelsize=8)
g_journal.set_axis_labels(x_var='Journal', y_var='Publications')
g_journal.set(title='Publications per journal')
