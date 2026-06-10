#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return SAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% 1- 260604_NF-cit_260602_AULu_ESAR_c_Temp
ciclos_AULu = glob("../260604_NF@cit_260602_AULu_ESAR_c_Temp/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_AULu = glob("../260604_NF@cit_260602_AULu_ESAR_c_Temp/**/*resultados.txt",recursive=True)

ciclos_AULu.sort()
resultados_AULu.sort()
conc_AULu =  20 #g/L

for p in ciclos_AULu:
    print('  ',p)
print('\n')
for res in resultados_AULu:
    print('  ',res)
print('-'*50)

SAR_AULu, tau_AULu, Hc_AULu = extraer_SAR_tau(resultados_AULu)
res_AULu=[]

#% ploteo ciclos
fig00, ax =plt.subplots(figsize=(8,6),constrained_layout=True,sharey=True,sharex=True)
ax.set_ylabel('M (A/m)')
for i,e in enumerate(ciclos_AULu):
    if '152dA' in e:
        _,_,_, H_AULu,M_AULu,_ = lector_ciclos(ciclos_AULu[i])
        ax.plot(H_AULu/1000,M_AULu,'-',label=f'{SAR_AULu[i]:.3uS}')
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 260527 AULu\n300 kHz & 58 kA/m')

print('Resultados AULu', '='*80,'\n')
for r in resultados_AULu:
    res_AULu.append(ResultadosESAR(os.path.dirname(r)))

rates_AULu = []
fig01, ax =plt.subplots(figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)
for i,r in enumerate(res_AULu):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_AULu.append(rate)
    print(f'WRate = {rate:.2f} °C/s')
    ax.plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='Warming Rate')
plt.suptitle(f'Templogs NF@cit 260527 AULu\n300 kHz & 58 kA/m')
#%% 2- 260604_NF-cit_260602_AUN_dil_ESAR_c_Temp
ciclos_AUNdil = glob("../260604_NF@cit_260602_AUN_dil_ESAR_c_Temp/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_AUNdil = glob("../260604_NF@cit_260602_AUN_dil_ESAR_c_Temp/**/*resultados.txt",recursive=True)

ciclos_AUNdil.sort()
resultados_AUNdil.sort()
conc_AUNdil =  13 #g/L

for p in ciclos_AUNdil:
    print('  ',p)
print('\n')
for res in resultados_AUNdil:
    print('  ',res)
print('-'*50)

SAR_AUNdil, tau_AUNdil, Hc_AUNdil = extraer_SAR_tau(resultados_AUNdil)
res_AUNdil=[]
#% ploteo ciclos
fig20, ax =plt.subplots(figsize=(8,6),constrained_layout=True,sharey=True,sharex=True)
ax.set_ylabel('M (A/m)')
for i,e in enumerate(ciclos_AUNdil):
    if '152dA' in e:
        _,_,_, H_AUNdil,M_AUNdil,_ = lector_ciclos(ciclos_AUNdil[i])
        ax.plot(H_AUNdil/1000,M_AUNdil,'-',label=f'{SAR_AUNdil[i]:.3uS}')
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 260527 AUNdil\n300 kHz & 58 kA/m')

print('Resultados AUNdil', '='*80,'\n')
for r in resultados_AUNdil:
    res_AUNdil.append(ResultadosESAR(os.path.dirname(r)))

rates_AUNdil = []
fig21, ax =plt.subplots(figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)
for i,r in enumerate(res_AUNdil):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_AUNdil.append(rate)
    print(f'WRate = {rate:.2f} °C/s')
    ax.plot(r.time,r.temperatura,'.-',label=f'{SAR_AUNdil[i]:.3uS}')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Templogs NF@cit 260527 AUNdil\n300 kHz & 58 kA/m')
#%% 3- 260604_NF-cit_260602_AUV_ESAR_c_Temp
ciclos_AUV = glob("../260604_NF@cit_260602_AUV_ESAR_c_Temp/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_AUV = glob("../260604_NF@cit_260602_AUV_ESAR_c_Temp/**/*resultados.txt",recursive=True)

ciclos_AUV.sort()
resultados_AUV.sort()
conc_AUV =  23 #g/L

for p in ciclos_AUV:
    print('  ',p)
print('\n')
for res in resultados_AUV:
    print('  ',res)
print('-'*50)
SAR_AUV, tau_AUV, Hc_AUV = extraer_SAR_tau(resultados_AUV)

#% ploteo
fig30, ax =plt.subplots(figsize=(8,6),constrained_layout=True,sharey=True,sharex=True)
ax.set_ylabel('M (A/m)')
for i,e in enumerate(ciclos_AUV):
    if '152dA' in e:
        _,_,_, H_AUV,M_AUV,_ = lector_ciclos(ciclos_AUV[i])
        ax.plot(H_AUV/1000,M_AUV,'-',label=f'{SAR_AUV[i]:.2uS}')
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 260527 AUV\n300 kHz & 58 kA/m')

res_AUV=[]
print('Resultados AUV', '='*80,'\n')
for r in resultados_AUV:
    res_AUV.append(ResultadosESAR(os.path.dirname(r)))
rates_AUV = []
fig31, ax =plt.subplots(figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)
for i,r in enumerate(res_AUV):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_AUV.append(rate)
    print(f'WRate = {rate:.2f} °C/s')
    ax.plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='Warming Rate')
plt.suptitle(f'Templogs NF@cit 260527 AUV\n300 kHz & 58 kA/m')

#%% 4- 260604_NF-cit_260527_ESAR_c_Temp_repeticion
ciclos_NF0527     = glob("../260604_NF@cit_260527_ESAR_c_Temp_repeticion/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_NF0527 = glob("../260604_NF@cit_260527_ESAR_c_Temp_repeticion/**/*resultados.txt",recursive=True)

ciclos_NF0527.sort()
resultados_NF0527.sort()
conc_NF0527 =  19.8 #g/L

for p in ciclos_NF0527:
    print('  ',p)
print('\n')
for res in resultados_NF0527:
    print('  ',res)
print('-'*50)
SAR_NF0527, tau_NF0527, Hc_NF0527 = extraer_SAR_tau(resultados_NF0527)

#% ploteo
fig40, ax =plt.subplots(figsize=(8,6),constrained_layout=True,sharey=True,sharex=True)
ax.set_ylabel('M (A/m)')
for i,e in enumerate(ciclos_NF0527):
    if '152dA' in e:
        _,_,_, H_NF0527,M_NF0527,_ = lector_ciclos(ciclos_NF0527[i])
        ax.plot(H_NF0527/1000,M_NF0527,'-',label=f'{SAR_NF0527[i]:.2uS}')
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 260527 NF0527\n300 kHz & 58 kA/m')


res_NF0527=[]
print('Resultados NF0527', '='*80,'\n')
for r in resultados_NF0527:
    res_NF0527.append(ResultadosESAR(os.path.dirname(r)))
rates_NF0527 = []
fig41, ax =plt.subplots(figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)
for i,r in enumerate(res_NF0527):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_NF0527.append(rate)
    print(f'WRate = {rate:.2f} °C/s')
    ax.plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='Warming Rate')
plt.suptitle(f'Templogs NF@cit 260527 NF0527\n300 kHz & 58 kA/m')
#%% 5- Normalizo ciclos por concentracion y ploteo comparativo

fig50,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(18,5),constrained_layout=True,sharey=True)

ax1.set_title(f'AULu\n{conc_AULu} g/L')
ax2.set_title(f'AUN diluida\n{conc_AUNdil} g/L')
ax3.set_title(f'AUV\n{conc_AUV} g/L')
ax4.set_title(f'NF0527\n{conc_NF0527} g/L')

for i,_ in enumerate(ciclos_AULu):
    _,_,_, H_AULu,M_AULu,_ = lector_ciclos(ciclos_AULu[i])
    ax1.plot(H_AULu/1000,M_AULu/conc_AULu,'-',label=f'{SAR_AULu[i]:.3uS}')

for j,_ in enumerate(ciclos_AUNdil):
    _,_,_, H_AUNdil,M_AUNdil,_ = lector_ciclos(ciclos_AUNdil[j])
    ax2.plot(H_AUNdil/1000,M_AUNdil/conc_AUNdil,'-',label=f'{SAR_AUNdil[j]:.3uS}')

for k,_ in enumerate(ciclos_AUV):
    _,_,_, H_AUV,M_AUV,_ = lector_ciclos(ciclos_AUV[k])
    ax3.plot(H_AUV/1000,M_AUV/conc_AUV,'-',label=f'{SAR_AUV[k]:.3uS}')

for m,_ in enumerate(ciclos_NF0527):
    _,_,_, H_NF0527,M_NF0527,_ = lector_ciclos(ciclos_NF0527[m])
    ax4.plot(H_NF0527/1000,M_NF0527/conc_NF0527,'-',label=f'{SAR_NF0527[m]:.3uS}')

ax1.set_ylabel('M/[NPM] (Am²/kg)')
for a in [ax1,ax2,ax3,ax4]:
    a.set_xlabel('H (kA/m)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')

#%% 5.1 ploteo comparativo de ciclos normalizados, excepto AULu

fig51,(ax2,ax3,ax4)=plt.subplots(1,3,figsize=(15,5),constrained_layout=True,sharey=True)


ax2.set_title(f'AUN diluida\n{conc_AUNdil} g/L')
ax3.set_title(f'AUV\n{conc_AUV} g/L')
ax4.set_title(f'NF0527\n{conc_NF0527} g/L')

for i,_ in enumerate(ciclos_AULu):
    _,_,_, H_AULu,M_AULu,_ = lector_ciclos(ciclos_AULu[i])
    ax1.plot(H_AULu/1000,M_AULu/conc_AULu,'-',label=f'{SAR_AULu[i]:.3uS}')

for j,_ in enumerate(ciclos_AUNdil):
    _,_,_, H_AUNdil,M_AUNdil,_ = lector_ciclos(ciclos_AUNdil[j])
    ax2.plot(H_AUNdil/1000,M_AUNdil/conc_AUNdil,'-',label=f'{SAR_AUNdil[j]:.3uS}')

for k,_ in enumerate(ciclos_AUV):
    _,_,_, H_AUV,M_AUV,_ = lector_ciclos(ciclos_AUV[k])
    ax3.plot(H_AUV/1000,M_AUV/conc_AUV,'-',label=f'{SAR_AUV[k]:.3uS}')

for m,_ in enumerate(ciclos_NF0527):
    _,_,_, H_NF0527,M_NF0527,_ = lector_ciclos(ciclos_NF0527[m])
    ax4.plot(H_NF0527/1000,M_NF0527/conc_NF0527,'-',label=f'{SAR_NF0527[m]:.3uS}')

ax2.set_ylabel('M/[NPM] (Am²/kg)')
for a in [ax2,ax3,ax4]:
    a.set_xlabel('H (kA/m)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')


#%% ploteo comparativo de errorbars de ESAR
cuadro = '$f=300$ kHz\n$H_0=58$ kA/m'
categorias = ['260602\nAULu', '260602\nAUN diluida', '260602\nAUV', '260527\nNF0527']
x = np.arange(len(categorias))

fig1, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

sep = 0.25

for i,s in enumerate(SAR_AULu):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_AUNdil):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for k,s in enumerate(SAR_AUV):
    ax.bar(k*sep+7*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')

for m,s in enumerate(SAR_NF0527):
    ax.bar(m*sep+11*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C3')

ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel('ESAR (W/g)')
ax.set_title('ESAR - Comparativa')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes,
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()
#%% ploteo comparativo de errorbars de tau
fig2, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

sep = 0.25

for i,s in enumerate(tau_AULu):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')
for j,s in enumerate(tau_AUNdil):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for k,s in enumerate(tau_AUV):
    ax.bar(k*sep+7*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')
for m,s in enumerate(tau_NF0527):
    ax.bar(m*sep+11*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C3')
ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel(r'$\tau$ (ns)')
#ax.set_xlabel('Categoría')
ax.set_title(r'$\tau$ - Comparativa')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes,
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()
#%% Idem Hc
fig3, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

sep = 0.25
for i,s in enumerate(Hc_AULu):
    ax.bar(i*sep-1*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')
for j,s in enumerate(Hc_AUNdil):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for k,s in enumerate(Hc_AUV):
    ax.bar(k*sep+7*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')
for m,s in enumerate(Hc_NF0527):
    ax.bar(m*sep+11*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C3')

ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel('H$_c$ (kA/m)')
ax.set_title('H$_c$ - Comparativa ')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes,
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()

#%% Todos los templogs
WR_AULu = ufloat(np.mean(rates_AULu), np.std(rates_AULu))
WR_AUNdil = ufloat(np.mean(rates_AUNdil), np.std(rates_AUNdil))
WR_AUV = ufloat(np.mean(rates_AUV), np.std(rates_AUV))
WR_NF0527 = ufloat(np.mean(rates_NF0527), np.std(rates_NF0527))

fig4, ax = plt.subplots(figsize=(9,5),constrained_layout=True)
t1,T1=[],[]
for i,r in enumerate(res_AULu):
    t1.append(r.time)
    T1.append(r.temperatura)
    ax.plot(r.time,r.temperatura,'.-',c='C0',label=f'AULu - {WR_AULu:.1uS} °C/s' if i ==0 else '')
t2,T2=[],[]
for j,r in enumerate(res_AUNdil):
    t2.append(r.time)
    T2.append(r.temperatura)
    ax.plot(r.time,r.temperatura,'.-',c='C1',label=f'AUNdil - {WR_AUNdil:.1uS} °C/s' if j ==0 else '')
t3,T3=[],[]
for k,r in enumerate(res_AUV):
    t3.append(r.time)
    T3.append(r.temperatura)
    ax.plot(r.time,r.temperatura,'.-',c='C2',label=f'AUV - {WR_AUV:.1uS} °C/s' if k ==0 else '')
t4,T4=[],[]
for m,r in enumerate(res_NF0527):
    t4.append(r.time)
    T4.append(r.temperatura)
    ax.plot(r.time,r.temperatura,'.-',c='C3',label=f'NF0527 - {WR_NF0527:.1uS} °C/s' if m ==0 else '')

ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax.grid()
ax.legend(loc='lower right',frameon=True,shadow=True,title='Muestra - Warming Rate',ncol=1)
ax.set_title('Comparativa templogs - $f=300$ kHz  $H_0=58$ kA/m')

#%%
tt_1, TT_1, t_common_1, Tmin_1, Tmax_1, Tmean_1 = banda_temperatura(t1, T1)
tt_2, TT_2, t_common_2, Tmin_2, Tmax_2, Tmean_2 = banda_temperatura(t2, T2)
tt_3, TT_3, t_common_3, Tmin_3, Tmax_3, Tmean_3 = banda_temperatura(t3, T3)
tt_4, TT_4, t_common_4, Tmin_4, Tmax_4, Tmean_4 = banda_temperatura(t4, T4)

labels = ['AULu', 'AUN dil', 'AUV', 'NF0527']

fig5,(ax1,ax2) = plt.subplots(2,1,figsize=(9,8),constrained_layout=True,sharex=True)

for t, T in zip(tt_1, TT_1):
    ax1.plot(t, T, '--', c='C0',alpha=0.3)
ax1.fill_between(t_common_1, Tmin_1, Tmax_1,alpha=0.3,color='C0')
ax1.plot(t_common_1, Tmean_1,'C0-', lw=1.5, label=f'{labels[0]} - {WR_AULu:.1uS} °C/s')

for t, T in zip(tt_2, TT_2):
    ax1.plot(t, T, '--', c='C2',alpha=0.5)
ax1.fill_between(t_common_2, Tmin_2, Tmax_2,color='C2',alpha=0.3)
ax1.plot(t_common_2, Tmean_2,'C2', lw=1.5, label=f'{labels[1]} - {WR_AUNdil:.1uS} °C/s')

for t, T in zip(tt_3, TT_3):
    ax2.plot(t, T, '--', c='C1',alpha=0.5)
ax2.fill_between(t_common_3, Tmin_3, Tmax_3,alpha=0.3,color='C1')
ax2.plot(t_common_3, Tmean_3,'C1', lw=1.5, label=f'{labels[2]} - {WR_AUV:.1uS} °C/s')

for t, T in zip(tt_4, TT_4):
    ax2.plot(t, T, '--', c='C3',alpha=0.5)
ax2.fill_between(t_common_4, Tmin_4, Tmax_4,alpha=0.3,color='C3')
ax2.plot(t_common_4, Tmean_4,'C3', lw=1.5, label=f'{labels[3]} - {WR_NF0527:.1uS} °C/s')

for a in [ax1,ax2]:
    a.set_ylabel('T (°C)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='Muestra  -  Warming Rate')
ax2.set_xlabel('t (s)')
plt.suptitle('Comparativa templogs - $f=300$ kHz  $H_0=58$ kA/m')
plt.show()

#%% Salvo figuras
fig00.savefig('00_ciclos_promedio_AULu.png',dpi=300)
fig01.savefig('01_templogs_AULu.png',dpi=300)

fig20.savefig('02_ciclos_promedio_AUNdil.png',dpi=300)
fig21.savefig('03_templogs_AUNdil.png',dpi=300)

fig30.savefig('04_ciclos_promedio_AUV.png',dpi=300)
fig31.savefig('05_templogs_AUV.png',dpi=300)

fig40.savefig('06_ciclos_promedio_NF0527.png',dpi=300)
fig41.savefig('07_templogs_NF0527.png',dpi=300)

fig50.savefig('08_ciclos_promedio_normalizados.png',dpi=300)
fig51.savefig('09_ciclos_promedio_normalizados_bis.png',dpi=300)

fig1.savefig('10_comparativa_ESAR.png',dpi=300)
fig2.savefig('11_comparativa_tau.png',dpi=300)
fig3.savefig('12_comparativa_Hc.png',dpi=300)
fig4.savefig('13_comparativa_templogs.png',dpi=300)
fig5.savefig('14_comparativa_templogs_promedio.png',dpi=300)
# fig7.savefig('13_comparativa_templogs_promedio.png',dpi=300)
#%%    