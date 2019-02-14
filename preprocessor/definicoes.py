import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
from mesh_manager import MeshManager
import yaml

# parent_dir = os.path.dirname(__file__)
parent_dir = os.getcwd()
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join('parent_parent_dir', 'utils')

# import importlib.util
# spec = importlib.util.spec_from_file_location("pymoab_utils", utils_dir + '/pymoab_utils.py')
# utpy = importlib.util.module_from_spec(spec)
# print(dir(utpy))
# import pdb; pdb.set_trace()

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)


input_file = data_loaded['input_file']
ext_msh = input_file + '.msh'
M1=MeshManager(ext_msh)
os.chdir(parent_dir)
all_volumes=M1.all_volumes

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
l1=3
l2=9
# Posição aproximada de cada completação
Cent_weels=[[0.5, 0.5, 0.5],[17.5, 17.5, 17.5]]

# Distância, em relação ao poço, até onde se usa malha fina
r0=5

# Distância, em relação ao poço, até onde se usa malha intermediária
r1=7
#--------------fim dos parâmetros de entrada------------------------------------
print("")
print("INICIOU PRÉ PROCESSAMENTO")
tempo0_pre=time.time()
def Min_Max(e):
    verts = M1.mb.get_connectivity(e)     #Vértices de um elemento da malha fina
    coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
    xmin, xmax = coords[0][0], coords[0][0]
    ymin, ymax = coords[0][1], coords[0][1]
    zmin, zmax = coords[0][2], coords[0][2]
    for c in coords:
        if c[0]>xmax: xmax=c[0]
        if c[0]<xmin: xmin=c[0]
        if c[1]>ymax: ymax=c[1]
        if c[1]<ymin: ymin=c[1]
        if c[2]>zmax: zmax=c[2]
        if c[2]<zmin: zmin=c[2]
    return([xmin,xmax,ymin,ymax,zmin,zmax])

def lu_inv(M):
    L=M.shape[0]
    s=1000
    if L<s:
        tinv=time.time()
        LU=linalg.splu(M)
        inversa=csc_matrix(LU.solve(np.eye(M.shape[0])))
        print(time.time()-tinv,M.shape[0],"tempo de inversão, ordem")
    else:
        div=1
        for i in range(1,int(L/s)+1):
            if L%i==0:
                div=i
        l=int(L/div)
        ident=np.eye(l)
        zeros=np.zeros((l,l),dtype=int)
        tinv=time.time()
        LU=linalg.splu(M)
        print(div,M.shape[0],"Num divisões, Tamanho")
        for j in range(div):
            for k in range(j):
                try:
                    B=np.concatenate([B,zeros])
                except NameError:
                    B=zeros
            if j==0:
                B=ident
            else:
                B=np.concatenate([B,ident])
            for i in range(div-j-1):
                B=np.concatenate([B,zeros])
            if j==0:
                inversa=csc_matrix(LU.solve(B))
                del(B)
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))
                del(B)
        print(time.time()-tinv,M.shape[0],div,"tempo de inversão, ordem")
    return inversa

#--------------Definição das dimensões dos elementos da malha fina--------------
# Esse bloco deve ser alterado para uso de malhas não estruturadas
all_volumes=M1.all_volumes
# print("Volumes:",all_volumes)
verts = M1.mb.get_connectivity(all_volumes[0])     #Vértices de um elemento da malha fina
coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
xmin, xmax = coords[0][0], coords[0][0]
ymin, ymax = coords[0][1], coords[0][1]
zmin, zmax = coords[0][2], coords[0][2]
for c in coords:
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]
dx0, dy0, dz0 = xmax-xmin, ymax-ymin, zmax-zmin # Tamanho de cada elemento na malha fina
#-------------------------------------------------------------------------------
print("definiu dimensões")
# ----- Definição dos volumes que pertencem à malha fina e armazenamento em uma lista----
# finos -> Lista qua armazena os volumes com completação e também aqueles com distância (em relação ao centroide)
#aos volumes com completação menor que "r0"
finos=[]
intermediarios=[]
pocos_meshset=M1.mb.create_meshset()

# Determina se cada um dos elementos está a uma distância inferior a "r0" de alguma completação
# O quadrado serve para pegar os volumes qualquer direção
for e in all_volumes:
    centroid=M1.mtu.get_average_position([e])
    # Cent_wells -> Lista com o centroide de cada completação
    for c in Cent_weels:
        dx=(centroid[0]-c[0])**2
        dy=(centroid[1]-c[1])**2
        dz=(centroid[2]-c[2])**2
        distancia=dx+dy+dz
        if dx<r0**2 and dy<r0**2 and dz<r0**2:
            finos.append(e)
            if dx<dx0/4+.1 and dy<dy0/4+.1 and dz<dz0/4+.1:
                M1.mb.add_entities(pocos_meshset,[e])
        if distancia<r1**2 and dx<r1**2/2:
            intermediarios.append(e)
M1.mb.tag_set_data(M1.finos_tag, 0,pocos_meshset)

print("definiu volumes na malha fina")

pocos=M1.mb.get_entities_by_handle(pocos_meshset)
volumes_d = [pocos[0]]
volumes_n = [pocos[1]]

finos_meshset = M1.mb.create_meshset()

print("definiu poços")
#-------------------------------------------------------------------------------
#Determinação das dimensões do reservatório e dos volumes das malhas intermediária e grossa
for v in M1.all_nodes:       # M1.all_nodes -> todos os vértices da malha fina
    c=M1.mb.get_coords([v])  # Coordenadas de um nó
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]

Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin  # Dimensões do reservatório
#-------------------------------------------------------------------------------

# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
# O valor 0.01 é adicionado para corrigir erros de ponto flutuante
for i in range(int(Lx/l2+1.01)):    lx2.append(xmin+i*l2)
for i in range(int(Ly/l2+1.01)):    ly2.append(ymin+i*l2)
for i in range(int(Lz/l2+1.01)):    lz2.append(zmin+i*l2)

#-------------------------------------------------------------------------------
press = 100.0
vazao = 10.0
dirichlet_meshset = M1.mb.create_meshset()
neumann_meshset = M1.mb.create_meshset()

if M1.gravity == False:
    pressao = np.repeat(press, len(volumes_d))

# # colocar gravidade
elif M1.gravity == True:
    pressao = []
    z_elems_d = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes_d])
    delta_z = z_elems_d + Lz
    pressao = M1.gama*(delta_z) + press
###############################################
else:
    print("Defina se existe gravidade (True) ou nao (False)")

M1.mb.add_entities(dirichlet_meshset, volumes_d)
M1.mb.add_entities(neumann_meshset, volumes_n)
M1.mb.add_entities(finos_meshset, volumes_n + volumes_d)

#########################################################################################
#jp: modifiquei as tags para sparse
neumann=M1.mb.tag_get_handle("neumann", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
dirichlet=M1.mb.tag_get_handle("dirichlet", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
###############################################################################################

M1.mb.tag_set_data(neumann, volumes_n, np.repeat(1, len(volumes_n)))
M1.mb.tag_set_data(dirichlet, volumes_d, np.repeat(1, len(volumes_d)))

M1.mb.tag_set_data(M1.wells_neumann_tag, 0, neumann_meshset)
M1.mb.tag_set_data(M1.wells_dirichlet_tag, 0, dirichlet_meshset)
M1.mb.tag_set_data(M1.finos_tag, 0, finos_meshset)
M1.mb.tag_set_data(M1.press_value_tag, volumes_d, pressao)
M1.mb.tag_set_data(M1.vazao_value_tag, volumes_n, np.repeat(vazao, len(volumes_n)))

#-------------------------------------------------------------------------------
lxd2=[lx2[0]+l1/2]
if len(lx2)>2:
    for i in np.linspace((lx2[1]+lx2[2])/2,(lx2[-2]+lx2[-3])/2,len(lx2)-3):
        lxd2.append(i)
lxd2.append(lx2[-1]-l1/2)

lyd2=[ly2[0]+l1/2]
if len(ly2)>2:
    for i in np.linspace((ly2[1]+ly2[2])/2,(ly2[-2]+ly2[-3])/2,len(ly2)-3):
        lyd2.append(i)
lyd2.append(ly2[-1]-l1/2)

lzd2=[lz2[0]+l1/2]
if len(lz2)>2:
    for i in np.linspace((lz2[1]+lz2[2])/2,(lz2[-2]+lz2[-3])/2,len(lz2)-3):
        lzd2.append(i)
lzd2.append(lz2[-1]-l1/2)

print("definiu planos do nível 2")

# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(l2/l1)):   lx1.append(i*l1)
for i in range(int(l2/l1)):   ly1.append(i*l1)
for i in range(int(l2/l1)):   lz1.append(i*l1)

lxd1=[xmin+dx0/100]
for i in np.linspace(xmin+1.5*l1,xmax-1.5*l1,int((Lx-3*l1)/l1+1.1)):
    lxd1.append(i)
lxd1.append(xmin+Lx-dx0/100)

lyd1=[ymin+dy0/100]
for i in np.linspace(ymin+1.5*l1,ymax-1.5*l1,int((Ly-3*l1)/l1+1.1)):
    lyd1.append(i)
lyd1.append(ymin+Ly-dy0/100)

lzd1=[zmin+dz0/100]
for i in np.linspace(zmin+1.5*l1,zmax-1.5*l1,int((Lz-3*l1)/l1+1.1)):
    lzd1.append(i)
lzd1.append(xmin+Lz-dz0/100)

print("definiu planos do nível 1")
node=M1.all_nodes[0]
coords=M1.mb.get_coords([node])
min_dist_x=coords[0]
min_dist_y=coords[1]
min_dist_z=coords[2]
#-------------------------------------------------------------------------------
'''
#----Correção do posicionamento dos planos que definem a dual
# Evita que algum nó pertença a esses planos e deixe as faces da dual descontínuas
for i in range(len(lxd1)):
    for j in range(len(lyd1)):
        for k in range(len(lzd1)):
            for n in range(len(M1.all_nodes)):
                coord=M1.mb.get_coords([M1.all_nodes[n]])
                dx=lxd1[i]-coord[0]
                dy=lyd1[j]-coord[1]
                dz=lzd1[k]-coord[2]
                if np.abs(dx)<0.0000001:
                    print('Plano x = ',lxd1[i],'corrigido com delta x = ',dx)
                    lxd1[i]-=0.000001
                    i-=1
                if np.abs(dy)<0.0000001:
                    print('Plano y = ',lyd1[j],'corrigido com delta y = ',dy)
                    lyd1[j]-=0.000001
                    j-=1
                if np.abs(dz)<0.0000001:
                    print('Plano z = ',lzd1[k],'corrigido dom delta z = ', dz)
                    lzd1[k]-=0.000001
                    k-=1
#-------------------------------------------------------------------------------
print("corrigiu planos do nível 1")'''
t0=time.time()
# ---- Criação e preenchimento da árvore de meshsets----------------------------
# Esse bloco é executado apenas uma vez em um problema bifásico, sua eficiência
# não é criticamente importante.
L2_meshset=M1.mb.create_meshset()       # root Meshset
D2_meshset=M1.mb.create_meshset()

###########################################################################################
#jp:modifiquei as tags abaixo para sparse
D1_tag=M1.mb.tag_get_handle("d1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
D2_tag=M1.mb.tag_get_handle("d2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################

fine_to_primal1_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
fine_to_primal2_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
AV_meshset=M1.mb.create_meshset()

primal_id_tag1 = M1.mb.tag_get_handle("PRIMAL_ID_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
primal_id_tag2 = M1.mb.tag_get_handle("PRIMAL_ID_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
nc1=0
nc2 = 0
lc1 = {}
#lc1 é um mapeamento de ids no nivel 1 para o meshset correspondente
#{id:meshset}

dict_m = {}

for e in all_volumes: M1.mb.add_entities(AV_meshset,[e])
for i in range(len(lx2)-1):
    t1=time.time()
    for j in range(len(ly2)-1):
        for k in range(len(lz2)-1):
            volumes_nv2 = []
            l2_meshset=M1.mb.create_meshset()
            d2_meshset=M1.mb.create_meshset()
            all_volumes=M1.mb.get_entities_by_handle(AV_meshset)
            for elem in all_volumes:
                centroid=M1.mtu.get_average_position([elem])
                if (centroid[0]>lx2[i]) and (centroid[0]<ly2[i]+l2) and (centroid[1]>ly2[j])\
                and (centroid[1]<ly2[j]+l2) and (centroid[2]>lz2[k]) and (centroid[2]<lz2[k]+l2):
                    M1.mb.add_entities(l2_meshset,[elem])
                    M1.mb.remove_entities(AV_meshset,[elem])
                    elem_por_L2=M1.mb.get_entities_by_handle(l2_meshset)

                if i<(len(lxd2)-1) and j<(len(lyd2)-1) and k<(len(lzd2)-1):
                    if (centroid[0]>lxd2[i]-l1/2) and (centroid[0]<lxd2[i+1]+l1/2) and (centroid[1]>lyd2[j]-l1/2)\
                    and (centroid[1]<lyd2[j+1]+l1/2) and (centroid[2]>lzd2[k]-l1/2) and (centroid[2]<lzd2[k+1]+l1/2):

                        M1.mb.add_entities(d2_meshset,[elem])
                        f1a2v3=0
                        if (centroid[0]-lxd2[i])**2<l1**2/4 or (centroid[0]-lxd2[i+1])**2<l1**2/4 :
                            f1a2v3+=1
                        if (centroid[1]-lyd2[j])**2<l1**2/4 or (centroid[1]-lyd2[j+1])**2<l1**2/4:
                            f1a2v3+=1
                        if (centroid[2]-lzd2[k])**2<l1**2/4 or (centroid[2]-lzd2[k+1])**2<l1**2/4:
                            f1a2v3+=1
                        M1.mb.tag_set_data(D2_tag, elem, f1a2v3)
                        M1.mb.tag_set_data(fine_to_primal2_classic_tag, elem, nc2)
            M1.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg=M1.mb.get_entities_by_handle(l2_meshset)
            # boundary_faces = utpy.get_boundary_of_volumes(M1.mb, sg)
            # meshset_faces = M1.mb.create_meshset()
            # M1.mb.add_entities(meshset_faces, boundary_faces)
            # print(boundary_faces)
            # import pdb; pdb.set_trace()



            print(k, len(sg), time.time()-t1)
            t1=time.time()
            d1_meshset=M1.mb.create_meshset()

            M1.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)
            nc2+=1

            for m in range(len(lx1)):
                a=l1*i+m
                for n in range(len(ly1)):
                    b=l1*j+n
                    for o in range(len(lz1)):
                        c=l1*k+o
                        l1_meshset=M1.mb.create_meshset()
                        for e in elem_por_L2:
                            # Refactory here
                            # Verificar se o uso de um vértice reduz o custo
                            centroid=M1.mtu.get_average_position([e])
                            if (centroid[0]>lx2[i]+lx1[m]) and (centroid[0]<lx2[i]+lx1[m]+l1)\
                            and (centroid[1]>ly2[j]+ly1[n]) and (centroid[1]<ly2[j]+ly1[n]+l1)\
                            and (centroid[2]>lz2[k]+lz1[o]) and (centroid[2]<lz2[k]+lz1[o]+l1):
                                M1.mb.add_entities(l1_meshset,[e])
                            if a<(len(lxd1)-1) and b<(len(lyd1)-1) and c<(len(lzd1)-1):
                                if (centroid[0]>lxd1[a]-1.5*dx0) and (centroid[0]<lxd1[a+1]+1.5*dx0)\
                                and (centroid[1]>lyd1[b]-1.5*dy0) and (centroid[1]<lyd1[b+1]+1.5*dy0)\
                                and (centroid[2]>lzd1[c]-1.5*dz0) and (centroid[2]<lzd1[c+1]+1.5*dz0):
                                    M1.mb.add_entities(d1_meshset,[elem])
                                    f1a2v3=0
                                    M_M=Min_Max(e)
                                    if (M_M[0]<lxd1[a] and M_M[1]>lxd1[a]) or (M_M[0]<lxd1[a+1] and M_M[1]>lxd1[a+1]):
                                        f1a2v3+=1
                                    if (M_M[2]<lyd1[b] and M_M[3]>lyd1[b]) or (M_M[2]<lyd1[b+1] and M_M[3]>lyd1[b+1]):
                                        f1a2v3+=1
                                    if (M_M[4]<lzd1[c] and M_M[5]>lzd1[c]) or (M_M[4]<lzd1[c+1] and M_M[5]>lzd1[c+1]):
                                        f1a2v3+=1
                                    M1.mb.tag_set_data(D1_tag, e,f1a2v3)
                                    M1.mb.tag_set_data(fine_to_primal1_classic_tag, e, nc1)


                        M1.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                        nc1+=1
                        M1.mb.add_child_meshset(l2_meshset,l1_meshset)
#-------------------------------------------------------------------------------
print('Criação da árvore: ',time.time()-t0)
ta=time.time()
all_volumes=M1.all_volumes

vert_meshset=M1.mb.create_meshset()

for e in all_volumes:
    d1_tag = int(M1.mb.tag_get_data(D1_tag, e, flat=True))
    if d1_tag==3:
        M1.mb.add_entities(vert_meshset,[e])
all_vertex_d1=M1.mb.get_entities_by_handle(vert_meshset)
mm=0
for x in lxd1:
    for y in lyd1:
        for z in lzd1:
            v1 = all_vertex_d1[0]
            c=M1.mtu.get_average_position([v1])
            d=(c[0]-x)**2+(c[1]-y)**2+(c[2]-z)**2
            for e in all_vertex_d1:
                c=M1.mtu.get_average_position([e])
                dist=(c[0]-x)**2+(c[1]-y)**2+(c[2]-z)**2
                if dist<d:
                    d=dist
                    v1=e
            M1.mb.tag_set_data(D1_tag, v1, 4)

for e in all_vertex_d1:
    d1_tag = int(M1.mb.tag_get_data(D1_tag, e, flat=True))
    if d1_tag==3:
        M1.mb.tag_set_data(D1_tag, e, 2)
    elif d1_tag==4:
        M1.mb.tag_set_data(D1_tag, e, 3)



print(time.time()-ta,"correção")
print("TEMPO TOTAL DE PRÉ PROCESSAMENTO:",time.time()-tempo0_pre)
print(" ")
os.chdir(flying_dir)
ext_h5m = input_file + '.h5m'
ext_vtk = input_file + '.vtk'
M1.mb.write_file(ext_h5m)
av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)
M1.mb.write_file(ext_vtk,[av])
