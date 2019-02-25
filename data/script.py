import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
import scipy.interpolate
import scipy.optimize
plt.rc('text', usetex = True)
plt.rc('font', size=13, family = 'serif')
# plt.rc('text.latex',unicode=True)
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')
import numpy as np
import math



fid1=open('data/data1.txt','r')
lines1=fid1.readlines()
x=[] 
I=[]
for i in range(0,len(lines1)):
    x.append(float(lines1[i].split(',')[0]))
    I.append(float(lines1[i].split(',')[1]))

Imax = np.amax(I)
I = np.array(I)/ Imax
x = np.array(x)

fitted = np.poly1d(np.polyfit(x,I,11))
xs = np.arange(0,4,0.1)

def func(Is,I,x):
    Is = Is / 178
    lam = 3.5*4
    inv_fitted = np.poly1d(np.polyfit(I[0:-2],x[0:-2],11))
    x = inv_fitted(Is)
    U = np.sin( 2*np.pi/lam*x )
    return U

inv_fitted = np.poly1d(np.polyfit(I[0:-2],x[0:-2],11))

# Is = np.arange(0,1,0.01)

# print('I_norm =',Is/178)
# print('U_norm =',func(Is,I,x))

def imp(K,dz):
    Zv = 50 
    C=  (2*np.pi)/( 3.5*4+1)
    return Zv*( 1j + K*np.tan( C * dz ) )/( 1j*K +np.tan( C*dz ) )

def gam(Z,Zv):
    return (Z-Zv)/(Z+Zv)

Z0 = 86
Is = 136
lam = 3.5*4+1


K = func(80,I,x)/func(10,I,x)
# print('1:',func(80,I,x))
# print('2:',func(10,I,x))
print('KСВ=',K)
Z_z=imp(func(136,I,x)/func(1,I,x),0.3)
Z_s=imp(func(162,I,x)/func(1,I,x),3.5)

print('Z_Закорот=',Z_z)
print('Z_Своб=',Z_s)
print('Z_Закорот_теор=',Z0*np.tan(2*np.pi*5/lam)*1j)
print('Z_Своб_теор=',-Z0/np.tan(2*np.pi*5/lam)*1j)
Zr= imp(K,1.9)
print('Z_r=',Zr)
print('GammaZ=',gam(Z_z,50))
print('GammaS=',gam(Z_s,50))
print('GammaR=',gam(Zr,50))



















# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# ax.plot(x,I,'ko')
# ax.plot(xs,fitted(xs),'k--')
# ax.plot(x,np.sin( 2*np.pi/lam*x ),'k-')
# # ax.plot(inv_fitted(Is),Is,'r--')

# ax.set_xticks(np.arange(0,4,0.1),minor = True)
# ax.set_yticks(np.arange(0,1,0.05),minor = True)
# ax.grid(True,which = 'both')
# plt.title(r'Градуировочный график')
# plt.xlabel(r'z, см')
# plt.ylabel(r'$I_{д}, U$')
# plt.legend( ['Измерения','Аппроксимация',r'$ \sin \frac{2 \pi }{\lambda} z $'],shadow = False, fancybox = False )
# plt.show()
# fig.savefig('graphs/grad.png',dpi=500)

