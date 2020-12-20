import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


def energy(state):
    sum = 0
    #rows
    for i in range(9):
        sum += np.exp(9 - len(set(state[i])))

    #columns and
    for i in range(9):
        sum += np.exp(9 - len(set(state[:,i])))
    
    #blocks
    for i in range(3):
        for j in range(3):
            # print(set(tuple(state[3*i:3*i + 3, 3*j:3*j + 3].reshape(1,-1).squeeze())))
            sum += np.exp(9 -len(set(tuple(state[3*i:3*i + 3, 3*j:3*j + 3].reshape(1,-1).squeeze()))))
    return sum


def move(state):
    initial_energy = energy(state)

    st = state.copy()
    st = st.reshape(1,-1).squeeze()

    a = np.random.randint(0, 81)
    b = np.random.randint(0, 81)

    new_state = state.copy()

    

    new_state = new_state.reshape(1,-1).squeeze()

    new_state[a], new_state[b] = st[b], st[a]

    new_state = new_state.reshape(9,9)

    return (energy(new_state) - initial_energy), new_state



def test_correct():

    grid = np.array([[7,2,6,4,9,3,8,1,5],
                    [3,1,5,7,2,8,9,4,6],
                    [4,8,9,6,5,1,2,3,7],
                    [8,5,2,1,4,7,6,9,3],
                    [6,7,3,9,8,5,1,2,4],
                    [9,4,1,3,6,2,7,5,8],
                    [1,9,4,8,3,6,5,7,2],
                    [5,6,7,2,1,4,3,8,9],
                    [2,3,8,5,7,9,4,6,1]])
    return grid

def test_incorrect():

    grid = np.array([[1,2,6,4,9,3,8,1,5],
                    [3,1,5,7,2,8,9,4,6],
                    [4,8,9,6,5,1,2,3,7],
                    [8,5,2,1,4,7,6,9,3],
                    [6,7,3,9,8,5,1,2,4],
                    [9,4,1,3,6,2,7,5,8],
                    [1,9,4,8,3,6,5,7,2],
                    [5,6,7,2,1,4,3,8,9],
                    [2,3,8,5,7,9,4,6,1]])
    return grid


def test_rand():
    grid = [i for i in range(1,10)]*9
    np.random.shuffle(grid)
    grid = np.array(grid).reshape(9,9)
    return grid



#initial_state = test_incorrect()
initial_state  = test_rand()

print(initial_state)




#initial state
print("Initial state")
print(initial_state)

print("Initial Energy = ", energy(initial_state))


# temperature
Tmax = 1
Tmin = 0.0001
Tfactor = -np.log(Tmax / Tmin)
steps = 2_000_000

T = Tmax

T_history = [T]


E = energy(initial_state)
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
    dE, cur_state  = move(cur_state)

    if dE is None:
        E = energy(cur_state)
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
    if E == 27:
        break

print("Best state:\n", best_state)
print("Best energy:\n", best_energy)

print("Final state:\n")
print(best_state)

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