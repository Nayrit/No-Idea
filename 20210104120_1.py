import random

POPULATION_SIZE = 50
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100

def generate_board_state():
    board_state = [random.randint(0, 7) for _ in range(8)]
    return board_state

def calculate_fitness(board_state):
    conflicts = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if board_state[i] == board_state[j] or abs(board_state[i] - board_state[j]) == j - i:
                conflicts += 1
    return 28 - conflicts  

def tournament_selection(population):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x[1])

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 7)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(board_state):
    pos1, pos2 = random.sample(range(8), 2)
    board_state[pos1], board_state[pos2] = board_state[pos2], board_state[pos1]
    return board_state

population = [(generate_board_state(), 0) for _ in range(POPULATION_SIZE)]

for generation in range(MAX_GENERATIONS):
    
    population = [(board_state, calculate_fitness(board_state)) for board_state, _ in population]

    best_board_state = max(population, key=lambda x: x[1])[0]
    if calculate_fitness(best_board_state) == 28:
        print("Solution found in generation", generation)
        break

    new_population = []

    new_population.append(max(population, key=lambda x: x[1]))

    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = crossover(parent1[0], parent2[0])
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        new_population.append((child, 0))

    population = new_population

print("Best solution:", best_board_state)