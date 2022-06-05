import pygame
import os
import random
import numpy

M=10                                        
N=10                                         
tamano_cuadricula=25                                
tamano_poblacion=500         
longitud_padres=50
archivo= 'generaciones.npz'           


def Snake_game():
    global Snake,screen,loop,mloop,pause_time
    pygame.init()
    screen = pygame.display.set_mode((M*tamano_cuadricula,N*tamano_cuadricula))
    (x1,y1)=random.choice(grids)
    (x2,y2)=random.choice([(x1,y1+1),(x1-1,y1),(x1,y1-1),(x1+1,y1)])
    Snake,steps,uniq,pause_time=[(x1,y1),(x2,y2)],0,[0]*(M*N-2),0
    comida()
    loop=True
    while loop:
        steps=steps+1
        prediccion_por_pesos_geneticos()
        actualizar_serpiente()
        if len(Snake)==M*N:
            print('Puntuacion Maxima')
            loop=False
        elif snake_head==Food:
            comida()
            Snake.append(snake_tail)
            pause_time =key_sensitive*5  if len(Snake)==M*N-10 else pause_time
        elif snake_head not in grids or snake_head in snake_body:
            loop=False
        ev=pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                pygame.quit()
                mloop,loop=False,False
            elif event.type == pygame.KEYDOWN:
                pause_time = pause_time+key_sensitive  if event.key == pygame.K_UP else pause_time-key_sensitive if event.key == pygame.K_DOWN and pause_time>=key_sensitive else pause_time
        if (Snake[0],Food) not in uniq:
            uniq.append((Snake[0],Food))
            del uniq[0]
        else:
            loop=False
    score=len(Snake)-2
    return (score+0.5+0.5*(score-steps/(score+1))/(score+steps/(score+1)))*1000000,score,steps
def comida():
    global Food
    snake_no_grids= [i for i in grids if i not in Snake]
    Food = random.choice(snake_no_grids)
def prediccion_por_pesos_geneticos():
    global action
    lstop=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    lst=[(0,-1),(1,0),(0,1),(-1,0)]
    head_diriction=lstop[lst.index((Snake[0][0]-Snake[1][0],Snake[0][1]-Snake[1][1]))]
    (x,y)=Snake[0]
    d1=[(x,_) for _ in range(y-1,-1,-1)]
    d3=[(_,y) for _ in range(x+1,M)]
    d5=[(x,_) for _ in range(y+1,N)]
    d7=[(_,y) for _ in range(x-1,-1,-1)]
    d8=[(x-_,y-_) for _ in range(1,min(x,y)+1)]
    d4=[(x+_,y+_) for _ in range(1,min(M-x,N-y))]
    d2=[(x+_,y-_) for _ in range(1,max(M-x,N-y)) if (x+_,y-_) in grids]
    d6=[(x-_,y+_) for _ in range(1,max(M-x,N-y))  if (x-_,y+_) in grids]
    d,val=[d1,d2,d3,d4,d5,d6,d7,d8],min(M,N)-1
    wall_distance=[len(i)/val for i in d]
    food_presence=[(val-j.index(Food))/val if Food in j else 0 for j in d]
    body_presence=[min([dv.index(v) if v in Snake else val for v in dv])/val if dv else 0 for dv in d]
    vision=[j[i] for i in range(8) for j in [wall_distance,body_presence,food_presence]]
    input_layer=vision+head_diriction
    action=red_neuronal(input_layer)
def red_neuronal(ip):
    m1,s1=numpy.reshape(ip,(1,NN[0])),0
    for _ in range(len(AF)):
        l1,l2=NN[_],NN[_+1]
        s2,s3=s1+l1*l2,s1+l2+l1*l2
        m2,m3=numpy.reshape(weights[s1:s2],(l1,l2)),numpy.reshape(weights[s2:s3],(1,l2))
        m4=numpy.matmul(m1,m2)+m3
        m1,s1=AF[_](m4),s3
    return Actions[numpy.argmax(m1)]
def relu(x):
    lst=[_ if _>0 else 0 for _ in x[0]]
    return numpy.array(lst)
def sigmoide(x):
    lst=[1/(1+(numpy.exp(-1*_))) for _ in x[0]]
    return numpy.array(lst)
def actualizar_serpiente():
    global snake_tail,snake_head,snake_body
    display()
    pygame.time.wait(pause_time)
    (x,y)=Snake[0]
    Snake.insert(0,(x+1,y)) if action == 'Right' else Snake.insert(0,(x-1,y)) if action == 'Left' else Snake.insert(0,(x,y+1)) if action == 'Bottum' else Snake.insert(0,(x,y-1))
    snake_tail=Snake.pop()
    snake_head,snake_body=Snake[0],Snake[1:]
def display():
    pygame.draw.rect(screen,(0,0,0), (0,0,M*tamano_cuadricula,N*tamano_cuadricula))
    pygame.draw.rect(screen,(255,255,255),(Snake[0][0]*tamano_cuadricula,Snake[0][1]*tamano_cuadricula,tamano_cuadricula,tamano_cuadricula))
    [pygame.draw.rect(screen,(255,255,255), (i[0]*tamano_cuadricula,i[1]*tamano_cuadricula,tamano_cuadricula,tamano_cuadricula),1) for i in Snake[1:]]
    pygame.draw.rect(screen,(0,255,0), (Food[0]*tamano_cuadricula,Food[1]*tamano_cuadricula,tamano_cuadricula,tamano_cuadricula))
    pygame.display.update()
def crossover():
    global offspring
    offspring=[]
    for _ in range(tamano_poblacion-longitud_padres):
        parant1_id=random.choice(Roulette_wheel)
        parant2_id=random.choice(Roulette_wheel)
        while parant2_id==parant1_id: parant2_id=random.choice(Roulette_wheel)
        wts=[parants[parant1_id][i] if random.uniform(0, 1) < 0.5 else parants[parant2_id][i] for i in range(weights_length)]
        offspring.append(wts)
def mutacion():
    global offspring
    for i in range(tamano_poblacion-longitud_padres):
        for _ in range(int(weights_length*0.05)):
            plc=random.randint(0,weights_length-1)
            value=random.choice(numpy.arange(-0.5,0.5,step=0.001))
            offspring[i][plc]=offspring[i][plc]+value

NN=[28,8,4]                                 
AF=[relu,sigmoide]                           
pause_time,key_sensitive,generation_length,mloop=0,15,2000,True
grids=[(i,j) for i in range(M) for j in range(N)]
Actions=['Top','Right','Bottum','Left']
Roulette_wheel=list(range(0,int(0.2*longitud_padres)))*3+list(range(int(0.2*longitud_padres),int(0.5*longitud_padres)))*2+list(range(int(0.5*longitud_padres),longitud_padres))
weights_length=sum([NN[_]*NN[_+1]+NN[_+1] for _ in range(len(NN)-1)])
if archivo not in os.listdir(os.getcwd()):
    population,statis=numpy.random.choice(numpy.arange(-1,1,step=0.001),size=(tamano_poblacion,weights_length),replace=True),numpy.array([0,0,0,0])
    Generacion,High_score=1,0
else:
    IP=numpy.load(archivo)
    population,statis=IP['POPULATION'],IP['STATIS']
    Generacion,High_score=statis[-1][0]+1,statis[-1][-1]


while Generacion<=generation_length and mloop:
    print('###################### ','Generacion ',Generacion,' ######################')
    Fitness,Score,i=[],[],0
    while i<tamano_poblacion and mloop:
        weights,i=list(population[i,:]),i+1
        fittness,score,steps=Snake_game()
        print('Cromosoma ',"{:03d}".format(i),' >>> ','Score : ',"{:03d}".format(score),', Pasos : ',"{:04d}".format(steps),', Fittness : ',fittness)
        Fitness.append(fittness),Score.append(score)
    parants,max_fittness,avg_score,j=[],max(Fitness),sum(Score)/len(Score),0
    while j<longitud_padres and mloop:
        j,parant_id=j+1,Fitness.index(max(Fitness))
        Fitness[parant_id]=-999
        parants.append(list(population[parant_id,:]))
    while mloop and j==longitud_padres:
        j=j+1
        High_score=max(Score) if max(Score) > High_score else High_score
        print('Puntuacion Mas alta de Generacion : ',max(Score),', Score Promedio de Generacion : ',avg_score,', Puntuacion General Mas Alta : ',High_score)
        crossover()
        mutacion()
        statis=numpy.row_stack((statis,numpy.array([Generacion,max(Score),avg_score,High_score])))
        population=numpy.reshape(parants+offspring,(tamano_poblacion,-1))
    Generacion=Generacion+1
pygame.quit()
numpy.savez(archivo,POPULATION=population,STATIS=statis)
