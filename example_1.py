import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# distance between the cities
def distance_l2(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)[0]

def distance_l1(a, b):
    return (abs(a[0]-b[0]) + abs(a[1] - b[1]))[0]

# distance is our energy
def energy(state, distance_matrix):
    e = 0
    for i in range(len(state)):
        e += distance_matrix[state[i-1]][state[i]]
    return e

def move(state, distance_matrix):
    initial_energy = energy(state, distance_matrix)

    a = np.random.randint(0, len(state) - 1)
    b = np.random.randint(0, len(state) - 1)

    new_state = state.copy()

    new_state[a], new_state[b] = state[b], state[a]

    return (energy(new_state, distance_matrix) - initial_energy), new_state

def test_1(n):
    cities = {}
    for i in range(N):
        x = 3*(i % 3)
        y = 3*(i // 3 ) % 9
        cities["city_" + str(i)] = (x + np.random.random(1),y + np.random.random(1))

    cities['city_0'] = (0,0)
    return cities

def test_2(n):
    cities = {}
    for i in range(N):
        
        cities["city_" + str(i)] = (10*np.random.random(1),10*np.random.random(1))

    cities['city_0'] = (0,0)
    return cities

N = 100

cities = test_1(N)

for k,v in cities.items():
    plt.plot(v[0],v[1],'o')
    plt.annotate(k,(v[0], v[1]))

#initial state 
initial_state =list(cities)
np.random.shuffle(initial_state)
print("Initial state")
print(initial_state)

#distance matrix
distance_matrix = defaultdict(dict)
for ka,va in cities.items():
    for kb,vb in cities.items():
        distance_matrix[ka][kb] = 0.0 if kb == ka else distance_l1(va, vb)


print("Initial Energy = ", energy(initial_state,distance_matrix))

# temperature
Tmax = 3
Tmin = 0.0001
Tfactor = -np.log(Tmax / Tmin)
steps = 2_00_000

T = Tmax

T_history = [T]

E = energy(initial_state, distance_matrix)
prev_energy = E
energy_history = [E]
best_energy = E

cur_state = initial_state.copy()
prev_state = cur_state

# annealing
step = 0
for step in tqdm(range(steps)):
    step += 1
    #T = Tmax * np.exp(Tfactor * step /steps)
    T = Tmax + step/steps*(Tmin - Tmax)
    T_history.append(T)
    dE, cur_state  = move(cur_state, distance_matrix)

    if dE is None:
        E = energy(cur_state, distance_matrix)
        dE = E - prev_energy
    else:
        E += dE

    # if we don't have improvement
    if dE > 0.0 and np.exp(-dE / T) < np.random.random():
        # Restore previous state
        cur_state = prev_state
        E = prev_energy
    else:
        # Accept new state and compare to best state
        prev_state = cur_state.copy()
        prev_energy = E
        if E < best_energy:
            best_state = cur_state.copy()
            best_energy = E

    energy_history.append(E)
    if(step % 100 == 0):
        print("current_energy = ",E)

while best_state[0] != 'city_0':
    best_state = best_state[1:] + best_state[:1]

print("Best state = ", best_state)
print("Best energy = ", best_energy)

print(" ➞  ".join(best_state))

x = []
y = []
for s in best_state:
    x.append(cities[s][0])
    y.append(cities[s][1])

plt.title("Route")
plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()

plt.figure()
plt.title("System energy")
plt.plot(energy_history)
plt.ylabel("Energy (distance)")
plt.xlabel("Iteration number")
plt.grid()

plt.figure()
plt.title("Temperature")
plt.plot(T_history)
plt.ylabel("Temperature")
plt.xlabel("Iteration number")
plt.grid()

plt.show()