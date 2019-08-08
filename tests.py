from neural_net_evo import Food, Organism, Environment


# Environment settings
x_min = -1
x_max = 1
y_min = -1
y_max = 1
tolerance = .075
num_orgs = 50
num_food = 100
num_gens = 50
gen_time = 100
dt = .04

# Food settings
food_value = 1

# Organism settings
v_max = .5
dv_max = .25
do_max = 720
n_input_nodes = 1
n_hidden_nodes = 5
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
	num_orgs=num_orgs,
	num_food=num_food,
	food_value=1,
	elitism=.20,
	num_gens=num_gens,
	gen_time=gen_time,
	dt=dt
	)

env.run()