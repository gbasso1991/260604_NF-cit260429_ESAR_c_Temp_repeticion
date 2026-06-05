
#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
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
#%% 1- 260604_NF-cit_260602_AULu_ESAR_c_Temp
ciclos_AULu = glob("../260604_NF-cit_260602_AULu_ESAR_c_Temp/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_AULu = glob("../260604_NF-cit_260602_AULu_ESAR_c_Temp/**/*resultados.txt",recursive=True)

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
plt.savefig('0_ciclos_promedio_NF@cit_260527.png',dpi=300)

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
ciclos_AUNdil = glob("../260604_NF-cit_260602_AUN_dil_ESAR_c_Temp/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_AUNdil = glob("../260604_NF-cit_260602_AUN_dil_ESAR_c_Temp/**/*resultados.txt",recursive=True)

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
plt.savefig('0_ciclos_promedio_NF@cit_260527.png',dpi=300)

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
ciclos_AUV = glob("../260604_NF-cit_260602_AUV_ESAR_c_Temp/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_AUV = glob("../260604_NF-cit_260602_AUV_ESAR_c_Temp/**/*resultados.txt",recursive=True)

ciclos_AUV.sort()
resultados_AUV.sort()
conc_AUV =  47 #g/L

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
plt.savefig('0_ciclos_promedio_NF@cit_260527.png',dpi=300)


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
ciclos_NF0527 = glob("../260604_NF-cit_260527_ESAR_c_Temp_repeticion/**/ciclo_promedio_H_M.txt",recursive=True)
resultados_NF0527 = glob("../260604_NF-cit_260527_ESAR_c_Temp_repeticion/**/*resultados.txt",recursive=True)

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
plt.savefig('0_ciclos_promedio_NF@cit_260527.png',dpi=300)


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
#%% Normalizo ciclos por concentracion y ploteo comparativo

fig5,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(18,5),constrained_layout=True,sharey=True)

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

ax1.set_ylabel('M/[NPM] (Am^2 /kg)')
for a in [ax1,ax2,ax3,ax4]:
    a.set_xlabel('H (kA/m)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')

#%%
# %% ploteo comparativo de errorbars de ESAR
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
    ax.bar(i*sep-2*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')
for j,s in enumerate(Hc_AUNdil):
    ax.bar(j*sep + 2*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for k,s in enumerate(Hc_AUV):
    ax.bar(k*sep+6*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')
for m,s in enumerate(Hc_NF0527):
    ax.bar(m*sep+10*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C3')

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

fig6, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

for i,r in enumerate(res_AULu):
    ax.plot(r.time,r.temperatura,'.-',c='C0',label=f'AULu - {WR_AULu:.1uS} °C/s' if i ==0 else '')

for j,r in enumerate(res_AUNdil):
    ax.plot(r.time,r.temperatura,'.-',c='C1',label=f'AUNdil - {WR_AUNdil:.1uS} °C/s' if j ==0 else '')
for k,r in enumerate(res_AUV):
    ax.plot(r.time,r.temperatura,'.-',c='C2',label=f'AUV - {WR_AUV:.1uS} °C/s' if k ==0 else '')
for m,r in enumerate(res_NF0527):
    ax.plot(r.time,r.temperatura,'.-',c='C3',label=f'NF0527 - {WR_NF0527:.1uS} °C/s' if m ==0 else '')

ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax.grid()
ax.legend(loc='lower right',frameon=True,shadow=True,title='Muestra - Warming Rate',ncol=1)
ax.set_title('Comparativa templogs - $f=300$ kHz  $H_0=58$ kA/m')
#%% Salvo figuras
fig00.savefig('0_ciclos_promedio_NF@cit_260527.png',dpi=300)
fig01.savefig('1_templogs_AULu_NF@cit_260527.png',dpi=300)
fig20.savefig('2_ciclos_promedio_AUNdil_NF@cit_260527.png',dpi=300)
fig21.savefig('3_templogs_AUNdil_NF@cit_260527.png',dpi=300)
fig30.savefig('4_ciclos_promedio_AUV_NF@cit_260527.png',dpi=300)
fig31.savefig('5_templogs_AUV_NF@cit_260527.png',dpi=300)
fig40.savefig('6_ciclos_promedio_NF0527_NF@cit_260527.png',dpi=300)
fig41.savefig('7_templogs_NF0527_NF@cit_260527.png',dpi=300)
fig5.savefig('8_ciclos_promedio_normalizados_NF@cit_260527.png',dpi=300)
fig1.savefig('9_comparativa_ESAR_NF@cit_260527.png',dpi=300)
fig2.savefig('10_comparativa_tau_NF@cit_260527.png',dpi=300)
fig3.savefig('11_comparativa_Hc_NF@cit_260527.png',dpi=300)
fig6.savefig('12_comparativa_templogs_NF@cit_260527.png',dpi=300)
