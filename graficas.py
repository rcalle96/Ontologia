# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:48:25 2024

@author: raul.calle
"""

# ==============================================================================
import os

# Tratamiento de df
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Herramientas
# ==============================================================================
import herramientas as h

import warnings
warnings.filterwarnings('ignore')


def calc_co2(data):
    co = [c for c in data.columns if 'CO_' in c][0]
    ch4 = [c for c in data.columns if 'CH4_' in c][0]
    c2h4 = [c for c in data.columns if 'C2H4_' in c][0]
    c2h6 = [c for c in data.columns if 'C2H6_' in c][0]
    c3h6 = [c for c in data.columns if 'C3H6_' in c][0]
    c3h8 = [c for c in data.columns if 'C3H8_' in c][0]
    c4h8 = [c for c in data.columns if 'C4H8_' in c][0]
    c4h10 = [c for c in data.columns if 'C4H10_' in c][0]
    c5h10 = [c for c in data.columns if 'C5H10_' in c][0]
    c5h12 = [c for c in data.columns if 'C5H12_' in c][0]
    ch3oh = [c for c in data.columns if 'CH3OH_' in c][0]
    c2h5oh = [c for c in data.columns if 'C2H5OH_' in c][0]
    ch3coh = [c for c in data.columns if 'CH3COH_' in c][0]
    hcooh = [c for c in data.columns if 'HCOOH_' in c][0]
    ch2o = [c for c in data.columns if 'CH2O_' in c][0]
    c2h4o2 = [c for c in data.columns if 'C2H4O2_' in c][0]
    nco2 = data[co] + data[ch4] + (2*data[c2h4]) + (2*data[c2h6]) + (3*data[c3h6]) + (3*data[c3h8]) + (4*data[c4h8]) + (4*data[c4h10]) + (5*data[c5h10]) + (5*data[c5h12]) + (data[ch3oh]) + (2*data[c2h5oh]) + (2*data[ch3coh]) + (data[hcooh]) + (data[ch2o]) + (2*data[c2h4o2])
    return nco2


def calc_selectividad_todos(data, compuestos=['H2', 'CO', 'CH4', 'CH3OH'],
                             ret_compuestos=False):
    new_data = data.copy()
    productos = [col for col in data.columns if '_mol_g' in col and col.endswith('g')]
    produccion_total = data[productos].sum(axis=1)
    total = 0
    for p in productos:
        reactivo = p.split('_')[0]
        sel = data.apply(lambda row: 100*row[p]/produccion_total[row.name] if produccion_total[row.name]>0 else 0, axis=1)
        total += sel.round(2)
        if reactivo in compuestos:
            new_data[reactivo] = sel.round(2)
    new_data['others'] = total - new_data[compuestos].sum(axis=1)
    if not ret_compuestos:
        return new_data
    else:
        return new_data, compuestos


def catalyst_family_classification(catalysts: pd.Series, family_list):
    # Convertir los catalizadores a serie de pandas
    if catalysts.__class__ is not pd.Series:
        catalysts = pd.Series(catalysts)
    # Diccionario vacio para almacenar los resultados
    dict_family = {}
    for cat in df_t['catalyst'].dropna().unique():
        # Valor por defecto, por si no se encuentra la familia
        family = 'other'
        for fam in cat_fam:
            if fam in cat:
                family = fam
                break
        dict_family[cat] = family
    return dict_family


def metricas_dispersion(serie, show_vals=True, ret_vals=False):
    mean = serie.mean()
    cv = 100*serie.std()/serie.mean()
    var = serie.var()
    sd = serie.sd()
    s_max = serie.meax()
    s_min = serie.min()
    rango = (s_min, s_max)
    if show_vals:
        print('Media:', mean)
        print('Varianza:', var)
        print('Desviación estándar:', sd)
        print(f'Rango: [{s_min}, {s_max}]')
        print('Coeficiente de Variacion:', cv)
    if ret_vals:
        return mean, var, sd, rango, cv


def metricas_dispersion_posicional(serie, q_low=0.25, q_high=0.75,
                                   show_vals=True, ret_vals=False):
    median = serie.median()
    q1, q3 = serie.quantile([q_low, q_high])
    iqr = q3 - q1
    if show_vals:
        print(f'Mediana: {median}')
        print(f'q bottom ({100*q_low}%): {q1}')
        print(f'q top ({100*q_high}%): {q3}')
        print(f'IQR: {iqr}')
    if ret_vals:
        return median, q1, q3, iqr


def add_quantile_reflines(fig, serie, qlow, qhigh):
    median, q_l, q_h, iqr = metricas_dispersion_posicional(serie, q_low=qlow, q_high=qhigh,
                                                           show_vals=False,
                                                           ret_vals=True)
    ql_label = str(100*qlow)
    qh_label = str(100*qhigh)
    fig.refline(y=median, color='g', alpha=1, label=f'median: {median:.2f}')
    for i, (q, limit) in enumerate(zip([q_l, q_h], [f'{ql_label}% limit', f'{qh_label}% limit'])):
        q_lim = q - ((1.5*iqr)*(-1)**i)
        label_lim = f'outliers: {">" if i==1 else "<"} {q_lim:.2f}'
        label = f'{limit}: {q:.2f}'
        fig.refline(y=q, color='k', label=label)
        fig.refline(y=q_lim, color='r', label=label_lim)


def porcentaje_columna(data, col):
    return 100*data[col].value_counts()/data[col].value_counts().sum()


def porcentaje_total(data, col):
    return 100*data[col].value_counts()/data.shape[0]


def plot_histogram(data, x, kde=True, shrink=.8, stat='percent',
                   label_rotation=0, label_size=10,
                   save_fig=False, carpeta=None,
                   *args, **kwargs):
    g = sns.displot(data=data, x=x, kind='hist', kde=True, shrink=shrink, stat=stat, *args, **kwargs)
    g.set(title=f'Columna: {x}')
    g.tick_params('x', labelrotation=label_rotation, labelsize=label_size)
    g.despine(top=False, right=False)
    if save_fig:
        #plt.subplots_adjust(top=0.9, bottom=-0.2)
        plt.tight_layout()
        if not carpeta:
            carpeta = h.seleccionar_carpeta()
        elif not os.path.isdir(carpeta):
            h.crear_carpeta(carpeta)
        else:
            plt.savefig(os.path.join(carpeta, x))
    plt.show()

ruta = 'C:/Users/raul.calle/Documents/RAUL/ONTOLOGIA'
ruta_figs = 'C:/Users/raul.calle/Documents/RAUL/ONTOLOGIA/Figuras BBDD'

# =============================================================================
# ---------------------- ENTRADAS DE LOS CATALIZADORES -------------------
# =============================================================================
# Ubicacion local
catadat_csv = "C://Users//raul.calle//Documents//RAUL//ONTOLOGIA//catalystsdata.csv"
# Nombres de columnas
column_names_cat = ['ID', 'no_ref', 'catalyst_name', 'TiO2_crystal_structure',
                    'catalyst', 'support', 'support_percent', 'co_catalyst',
                    'co_catalyst_2', 'co_catalyst_3', 'percent', 'percent_cc_2',
                    'percent_cc_3', 'dopant', 'dopant_percent', 'dyes', 'dye_percent',
                    'deposition_method', 'BET_m2_g', 'Eg_eV', 'comments',
                    'masscat_g', 'catalyst_set_up_ant', 'catalyst_set_up',
                    'reaction_medium', 'ph_value', 'illuminated_area_m2',
                    'reactor_volume_l', 'reactor_type_ant', 'reactor_type',
                    'operation_mode', 'reductant', 'CO2_H2O_reductant_ratio',
                    'T_C', 'P_bar', 'reaction_time_h', 'residence_time_min1',
                    'light_source', 'lamp', 'wavelength_nm', 'wavelength_nm_old',
                    'power_w', 'light_intensity_w_m2', 'H2_mol_gh', 'CO_mol_gh',
                    'CH4_mol_gh', 'C2H4_mol_gh', 'C2H6_mol_gh', 'C3H6_mol_gh',
                    'C3H8_mol_gh', 'C4H8_mol_gh', 'C4H10_mol_gh', 'C5H10_mol_gh',
                    'C5H12_mol_gh', 'CH3OH_mol_gh', 'C2H5OH_mol_gh', 'CH3COH_mol_gh',
                    'HCOOH_mol_gh', 'CH2O_mol_gh', 'C2H4O2_mol_gh', 'H2_mol_g',
                    'CO_mol_g', 'CH4_mol_g', 'C2H4_mol_g', 'C2H6_mol_g',
                    'C3H6_mol_g', 'C3H8_mol_g', 'C4H8_mol_g', 'C4H10_mol_g',
                    'C5H10_mol_g', 'C5H12_mol_g', 'CH3OH_mol_g', 'C2H5OH_mol_g',
                    'CH3COH_mol_g', 'HCOOH_mol_g', 'CH2O_mol_g', 'C2H4O2_mol_g',
                    'H2_mol_m2h', 'CO_mol_m2h', 'CH4_mol_m2h', 'C2H4_mol_m2h',
                    'C2H6_mol_m2h', 'C3H6_mol_m2h', 'C3H8_mol_m2h', 'C4H8_mol_m2h',
                    'C4H10_mol_m2h', 'C5H10_mol_m2h', 'C5H12_mol_m2h',
                    'CH3OH_mol_m2h', 'C2H5OH_mol_m2h', 'CH3COH_mol_m2h',
                    'HCOOH_mol_m2h', 'CH2O_mol_m2h', 'C2H4O2_mol_m2h', 'timestamp']
# Dataframe
df_cat = pd.read_csv(catadat_csv, names=column_names_cat)
# =============================================================================
# ---------------------- REFERENCIAS DE LOS ARTICULOS -------------------
# =============================================================================
# Ubicacion local
papers_csv = "C://Users//raul.calle//Documents//RAUL//ONTOLOGIA//paper_references (1).csv"
# Nombres de columnas
column_names_refs = ['ID', 'valid', 'no_ref', 'new_ref', 'filename', 'DOI', 'title',
                     'year', 'journal', 'pages', 'issue', 'abstract', 'authors',
                     'volume', 'corresponding_author', 'country', 'country_name']
# Dataframe
df_refs = pd.read_csv(papers_csv, names=column_names_refs)

# =============================================================================
# ------------------ TRANSFORMACION DE LOS DATOS DE LA TABLA -----------------
# =============================================================================
# Separar las variables categoricas y numericas
cols_obj = df_cat.select_dtypes('object').columns
cols_num = df_cat.select_dtypes(exclude='object').columns

# Columnas para de informacion y produccion
cols_info = [col for col in df_cat.columns if '_mol_' not in col]
cols_produccion = [col for col in df_cat.columns if '_mol_g' in col and col.endswith('g')]

# Limpiar los espacios al inicio y final de las cadenas de texto
df_t = df_cat.replace(r'^\s+|\s+$', '', regex=True)
for col in df_t:
    # -------------------- COLUMNAS CATEGORICAS ---------------------------
    if col in cols_obj:
        if df_t[col].str.contains(r'\d+').all():
            # Conversion de tipo a numerico
            df_t[col] = pd.to_numeric(df_t[col], errors='coerce')
            # Cambio de valores "-1" por "0"
            df_t[col].replace({-1: 0}, inplace=True)
        else:
            # Cambio de valores "-1" por "Not specified"
            df_t[col].replace({'-1': 'Not specified'})
    # -------------------- COLUMNAS NUMERICAS ---------------------------
    elif col in cols_num:
        if col in cols_produccion:
            # Cambio de valores "-1" por "0"
            df_t[col].replace({-1: 0}, inplace=True)

# Calculo de los moles equivalentes de CO2 convertidos en productos
df_t['nCO2'] = calc_co2(df_t[cols_produccion])
# Calculo de la produccion total (todos los productos) por cada catalizador
df_t['prod_total'] = df_t[cols_produccion].sum(axis=1)

# Definición de las familias principales de catalizadores
cat_fam = ['TiO2', 'ZnO', 'CeO2', 'MOF', 'LDH', 'C3N4']
# Clasificación de los catalizadores existentes en la tabla en su familia
# dict_fam = {cat: fam for cat in df_t['catalyst'].dropna().unique() for fam in cat_fam if fam in cat}
dict_fam = catalyst_family_classification(df_t['catalyst'], cat_fam)

# Incorporación de la columna <<cat_fam>> al DataFrame
df_t['cat_fam'] = df_t['catalyst'].map(dict_fam)

# Limpiar registros con conversiones equivalentes de CO2 nulas
df_f = df_t.loc[df_t['nCO2'] > 0, :]

# Estaditicos robustos (posicionales) para la dispersión
median_co2, ql_co2, qh_co2, iqr_co2 = metricas_dispersion_posicional(df_f['nCO2'],
                                                                     q_low=0.05, q_high=0.95,
                                                                     show_vals=False,
                                                                     ret_vals=True)
num_datos_90 = df_f.loc[(df_f['nCO2'] > ql_co2) & (df_f['nCO2'] < qh_co2)].shape[0]

# =============================================================================
# ------------------------ CONDICIONES FILTRADO --------------------------
# =============================================================================
# --------------------- Catalizadores freq acumulada > 66 ---------------------
cat_freq = porcentaje_total(df_f, 'catalyst')
main_cats = cat_freq[cat_freq.sort_values(ascending=False).cumsum() < 66].index

# --------------------- Catalizador en familia comun ---------------------
family = df_f['cat_fam'].notna()

# ------------------------ Catalizador TiO2 ------------------------------
cat_tio2 = df_f['catalyst'] == 'TiO2'

# ----------------------- Co-catalizador = NaN -------------------------
cocat_na = df_f['co_catalyst'].isna()

# ------------------------ Dopante = NaN ------------------------
dop_na = df_f['dopant'].isna()

# ------------------------ Dye = NaN ------------------------
dye_na = df_f['dyes'].isna()

# ------------------------ Support = NaN ------------------------
sup_na = df_f['support'].isna()

# -------------------- Numero de articulos por valor de CO2 -------------------
co2_paper_count = df_f.groupby('nCO2')['no_ref'].value_counts().reset_index()
df_count = co2_paper_count.merge(df_f[['no_ref', 'catalyst', 'cat_fam']])


# =============================================================================
# ------------------------------  GRAFICAS  --------------------------
# =============================================================================
# ------------------------------ HISTOGRAMAS ---------------------------------
carpeta_histogramas = "C:/Users/raul.calle/Documents/RAUL/ONTOLOGIA/Figuras BBDD/Histogramas 2"
for col in df_f.select_dtypes('object'):
    plot_histogram(df_f, x=col, label_rotation=90, aspect=2, shrink=.6)#,
                   # carpeta=carpeta_histogramas, save_fig=False)
for col in df_f.select_dtypes(exclude='object'):
    plot_histogram(df_f, x=col)#, carpeta=carpeta_histogramas, save_fig=False)


# ---------------------- publicaciones vs año -------------------------
df_refs_year = df_refs['year'].value_counts().reset_index().sort_values('year')
df_refs_year['cummulative'] = df_refs_year['count'].cumsum()

g_year = sns.catplot(data=df_refs_year, x='year', y='cummulative',
                     kind='bar', height=5, aspect=1.5, palette='turbo_r')
g_year.tick_params('x', labelsize=10, labelrotation=90)
g_year.set_axis_labels(x_var='Year', y_var='Publications')
g_year.despine(top=False, right=False)
g_year.set(title='Publications per year')


# ----------- catalizador vs año publicacion --------------------
df_year_cat = df_f[['no_ref', 'catalyst', 'cat_fam']].merge(df_refs[['no_ref', 'year']])

g_year_cat = sns.displot(data=df_year_cat, x='year', hue='cat_fam', kind='hist',
                         multiple='stack', discrete=True,
                         aspect=1.5, facet_kws={'legend_out': True})
g_year_cat.set(title='Publications per year per catalyst family')
g_year_cat.despine(top=False, right=False)
g_year_cat.legend.set_title(title='Catalyst family')
sns.move_legend(g_year_cat, loc='center right', bbox_to_anchor=(1.05, 0.5))


# ------------------------------ nCO2 vs TiO2 ---------------------------------
df_tio2 = df_f.loc[(cat_tio2 & cocat_na & dop_na & dye_na & sup_na)].sort_values(by='nCO2')

g_co2_tio2 = sns.catplot(data=df_tio2, x='catalyst', y='nCO2', kind='swarm', size=3)
add_quantile_reflines(g_co2_tio2, df_f['nCO2'], 0.05, 0.95)
g_co2_tio2.despine(top=False, right=False)
g_co2_tio2.set(title='CO2 equivalent mole conversion')
g_co2_tio2.set_axis_labels(x_var='', y_var='nCO2 (μmol/g)')
g_co2_tio2.set(yscale='log', ylim=(5e-3, 1e+6))
plt.legend()
sns.move_legend(g_co2_tio2.ax, loc=(1.01, 0.45))


# ---------------------------- nCO2 vs fam_cat --------------------------------
df_fam = df_f.loc[family].sort_values(by='nCO2')

g_fam = sns.catplot(data=df_fam, x='cat_fam', y='nCO2',
                    hue='cat_fam',
                    kind='swarm', size=2, aspect=2, legend=False)
add_quantile_reflines(g_fam, df_f['nCO2'], qlow=0.05, qhigh=0.95)
g_fam.despine(top=False, right=False)
g_fam.set(title='CO2 equivalent mole conversion')
g_fam.set_axis_labels(x_var='Catalyst Family', y_var='nCO2 (μmol/g)')
g_fam.set(yscale='log', ylim=(5e-3, 1e+6))
plt.legend()
sns.move_legend(g_fam.ax, loc=(1.01, 0.45))


# ----------- nCO2 vs masa catalizador -----------------------
df_m_co2 = df_f.query("masscat_g > 0")
g_co2_mass = sns.relplot(data=df_m_co2, x='masscat_g', y='nCO2', hue='cat_fam')
add_quantile_reflines(g_co2_mass, df_f['nCO2'], 0.05, 0.95)
g_co2_mass.set(xscale='log', yscale='log')
g_co2_mass.despine(top=False, right=False)


# ----------- selectividad vs catalizador ---------------------
# Calcular selectividad para compuestos (H2, CH4, CO, CH3OH)
df_sel_cat, compuestos = calc_selectividad_todos(df_f[cols_info + cols_produccion + ['prod_total', 'cat_fam']],
                                                 ret_compuestos=True)

df_melt_sel = df_sel_cat.melt(id_vars=cols_info + ['prod_total', 'cat_fam'],
                              value_vars=compuestos + ['others'],
                              var_name='product', value_name='selectivity')

# Histograma para comparar la selectividad de todos los catalizadores por cada producto
hist_sel = sns.displot(data=df_melt_sel,
                       x='selectivity', hue='product',
                       stat='percent', bins=10, multiple='stack', shrink=.7,
                       aspect=1.5, kind='hist')
hist_sel.despine(top=False, right=False)
hist_sel.set(title='Product selectivity')


# Histograma para comparar la selectividad de todos los catalizadores por cada
# producto, teniendo en cuenta solo los registros con una selectividad > 0
hist = sns.displot(data=df_melt_sel.query("selectivity > 0"),
                   x='selectivity',
                   hue='product', stat='percent', bins=10, shrink=.8,
                   multiple='stack', aspect=1.5, kind='hist')
hist.despine(top=False, right=False)
hist.set(title='Product selectivity if selectivity > 0')


# Histograma para comparar la selectividad de cada familia de catalizadores
# respecto a cada producto
g_hist = sns.displot(data=df_melt_sel,
                     col='product', #col_wrap=2,
                     hue='cat_fam', multiple='stack',
                     x='selectivity',
                     stat='percent',
                     bins=10)
g_hist.set_titles(col_template='{col_name} selectivity')
