import pygmsh
import numpy as np
from sympy import symbols, integrate, simplify
import matplotlib.pyplot as plt
from sympy import Matrix
import matplotlib.tri as tri
import time 

X=symbols('X')
Y=symbols('Y')

def u_analytical(x,y):    
    return (0.25)*(1-(x**2)-(y**2))

size = 2
with pygmsh.geo.Geometry() as geom:
    r = 1.0
    circle = geom.add_circle([0, 0, 0], r, mesh_size=size)
    mesh = geom.generate_mesh()

point = mesh.points
x= mesh.points[1:,0]
y= mesh.points[1:,1]
loca = mesh.cells_dict["triangle"]

#Determining Nx and Ny
Nx = len(np.unique(x))
print(f"Nx: {Nx}")
Ny = len(np.unique(y))
print(f"Ny: {Ny}")

#Defining size of global matrix and vector
n=len(point[:,0])-1

#Define global force vector 
K=np.zeros((n,n),dtype=float)

#Define global stiffness matrix
f=np.zeros((n,1),dtype=float)

#Building locals and adding to globals for each element
for e in range(len(loca[:,0])):

    #Extracting physical dimensions for each node
    nodes_element=loca[e]

    #Basis functions for triangle mesh
    phi = Matrix([1 - X - Y, X, Y])

    #Defining local stiffness matrix at each element 
    K_local=np.zeros((3,3),dtype=float)

    f_local=np.zeros((3,1),dtype=float)
    
    #Extracting physical dimensions from mesh
    coords=np.array([mesh.points[nodes_element[0]],mesh.points[nodes_element[1]],mesh.points[nodes_element[2]]])
    xk1,yk1=coords[0,:2]
    xk2,yk2=coords[1,:2]
    xk3,yk3=coords[2,:2]

    #Jacobian from physical dimensions
    Jac = np.array([[xk2 - xk1, xk3 - xk1],
                [yk2 - yk1, yk3 - yk1]])
    detJ = np.linalg.det(Jac)
    invJT = np.linalg.inv(Jac).T  # J^{-T}

    #Defining gradient
    gradient = np.array([[-1,-1],[1,0],[0,1]])

    #Defining gradient in physical dimensions
    gradient_X_Y = gradient@invJT.T

    #Determining area based on determininat
    area = abs(detJ)/2
    
    #Defining Local for each element
    for i in range(0,3): 
        for j in range(0,3): 
            K_local[i,j] = np.dot(gradient_X_Y[i], gradient_X_Y[j])*area
        aux2= simplify(phi[i]*float(abs(detJ)))
        f_local[i]=float(integrate(integrate(simplify(aux2), (X, 0, 1 - Y)), (Y, 0, 1)))

    #Adding locals to globals
    for i in range(3):
        for j in range(3):
            K[nodes_element[i]-1,nodes_element[j]-1]+=K_local[i,j]
        f[nodes_element[i]-1]+=f_local[i]

#Boundary Conditions
zeros =[]
aux=1e-6
i = 0
#Determining where boundary conditions should apply
for (x,y,_) in (mesh.points): 
    if abs(x**2+y**2-1)<aux: 
        zeros.append(i)
    i+=1

#Applying the boundary conditions
for i in zeros: 
    K[i-1,:]=0
    K[:,i-1]=0
    f[i-1]=0
    K[i-1,i-1]=1

#Backslash
ts_bs_1 = time.time()
u=np.linalg.solve(K,f)
ts_bs_2 = time.time()
print(f"Backslah Performance:{ts_bs_2-ts_bs_1}")

#Regular matrix multiplication 
ts_l_1=time.time()
u = np.linalg.inv(K)@f
ts_l_2 = time.time()
print(f"Linear Algebra Performance:{ts_l_2-ts_l_1}")

u =np.vstack([np.zeros((1,1)),u])

#Determning analytical solution
x_analytical =mesh.points[:,0]
y_analytical =mesh.points[:,1]
u_an=u_analytical(x_analytical,y_analytical)


#Processing u for plotting
u_flat=u.flatten()
x= mesh.points[:,0]
y= mesh.points[:,1]
triangle=tri.Triangulation(x,y,loca)

#Plotting FEA Approximation
plt.triplot(x, y, loca)
plt.gca().set_aspect("equal")
cont = plt.tricontourf(triangle,u_flat, levels=50, cmap="coolwarm")
plt.colorbar(cont)
plt.title(f"FEA Approximation, pygmsh mesh, mesh_size = {size}")
plt.xlabel('x')
plt.ylabel('y')

#Plotting Analytical Solution 
plt.figure(2)
triangle=tri.Triangulation(x,y,loca)

plt.triplot(x, y, loca)
plt.gca().set_aspect("equal")
cont2 = plt.tricontourf(triangle,u_an, levels=50, cmap="coolwarm")
plt.colorbar(cont2)
plt.title(f"Analytical Solution, pygmsh mesh, mesh_size = {size}")
plt.xlabel('x')
plt.ylabel('y')

plt.show()