
import numpy as np
#import numba as nb
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.widgets import Slider
nx=100
ny=100
nt=326
def csv_read_sfield(path):
	with open(path,'r',encoding='utf-8') as f:
		reader = csv.reader(f,dialect='excel')
		Z=[]
		for row in reader:
			Z.append([float(z) for z in row])

	X,Y=np.meshgrid(np.arange(nx), np.arange(ny))
	Z=np.array(Z)
	return X,Y,Z

def showsfield(u):

	fig = plt.figure()
	ax1 = fig.add_subplot()

	x1 = np.linspace(0, 1, nx)
	y1 = np.linspace(0, 1, nx)
	x1, y1 = np.meshgrid(x1, y1)
	ax1.contourf (x1, y1, u)
	c=ax1.contour (x1, y1, u,colors='black')
	plt.clabel(c,inline=True,fontsize=10)
	ax1.axis('equal')
	plt.show()
	
fig = plt.figure()
ax = fig.add_subplot(111)
axT = plt.axes([0.1,0.15,0.75,0.03])
sT =Slider(axT,'t',0,nt, valinit=0)
def update(event):
	ax.cla()
	
	X,Y,Z=csv_read_sfield(r'D:\NSEq\v_'+str(int(sT.val))+'.csv')
	x1 = np.linspace(0, 1, nx)
	y1 = np.linspace(0, 1, nx)
	x1, y1 = np.meshgrid(x1, y1)
	ax.contourf (x1, y1, Z)
	c=ax.contour (x1, y1, Z,colors='black')
	plt.clabel(c,inline=True,fontsize=10)
	ax.axis('equal')

sT.on_changed(update)

update(1)
plt.show()
