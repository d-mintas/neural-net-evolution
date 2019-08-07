from neural_net_evo import Food, Organism


# Environment settings
x_min = -1
x_max = 1
y_min = -1
y_max = 1
tol = .075

# Food settings
food_value = 1

# Organism settings
v_max = .5
dv_max = .25
do_max = 720
n_input_nodes = 1
n_hidden_nodes = 5
n_output_nodes = 2


food = Food(x_min, x_max, y_min, y_max, food_value)

org = Organism(
	x_min,
	x_max,
	y_min,
	y_max,
	v_max,
	dv_max,
	do_max,
	n_input_nodes,
	n_hidden_nodes,
	n_output_nodes,
	tol
	)