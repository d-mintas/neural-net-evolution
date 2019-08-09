from math import sqrt
import pygame
from neural_net_evo import Food, Organism, Environment


# Environment settings
x_min = 0.0
x_max = 4.0
y_min = 0.0
y_max = 4.0
tolerance = .075
elitism = .60
mutation_rate = 0.10
num_orgs = 50
num_food = 100
num_gens = 9999999999
gen_time = 100
dt = 0.04
colors = [
    (255, 0, 0), # red
    (0, 0, 255), # blue
    (0, 255, 0), # green
    # (255, 0, 255) # purple
]

# Food settings
food_value = 1

# Organism settings
v_max = 0.5
dv_max = 0.25
do_max = 720
n_input_nodes = 1
n_hidden_nodes = 2
n_output_nodes = 2


env = Environment(
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    v_max=v_max,
    dv_max=dv_max,
    do_max=do_max,
    n_input_nodes=n_input_nodes,
    n_hidden_nodes=n_hidden_nodes,
    n_output_nodes=n_output_nodes,
    tolerance=tolerance,
    mutation_rate=mutation_rate,
    num_orgs=num_orgs,
    num_food=num_food,
    food_value=food_value,
    elitism=elitism,
    num_gens=num_gens,
    gen_time=gen_time,
    dt=dt,
    colors=colors
    )

org_color = (0, 255, 127)
food_color = (105, 105, 105)
food_radius = 10

width, height = 600, 600
bg_color = (255, 255, 255)

x_scale = width / (x_max - x_min)
y_scale = width / (y_max - y_min)

screen = pygame.display.set_mode((width, height))
screen.fill(bg_color)

env_gen = env.run()

running = True
while running:
    screen.fill(bg_color)
    positions = next(env_gen)
    for food in positions['foods']:
        pygame.draw.rect(
            screen,
            food_color,
            (
                int((food[0] * x_scale) - (food_radius / 2)),
                int((food[1] * y_scale) - (food_radius / 2)),
                food_radius,
                food_radius
            ),
            3
        )
    for org in positions['orgs']:
        pygame.draw.circle(
            screen,
            org[3],
            (int(org[0] * x_scale), int(org[1] * y_scale)),
            int(sqrt(org[2] / 3) + 3),
            0
        )       
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False