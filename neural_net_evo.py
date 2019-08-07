import numpy as np


class Food(object):

    def __init__(self, x_min, x_max, y_min, y_max, food_value):
        self.x = np.random.uniform(x_min, x_max)
        self.y = np.random.uniform(y_min, y_max)
        self.value = food_value

    def respawn(self, x_min, x_max, y_min, y_max, food_value):
        self.x = np.random.uniform(x_min, x_max)
        self.y = np.random.uniform(y_min, y_max) 
        self.value = food_value


class Organism(object):

    def __init__(
        self,
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
        ):

        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.v_max = v_max
        self.dv_max = dv_max
        self.do_max = do_max
        self.tol = tol

        self.x = np.random.uniform(x_min, x_max)
        self.y = np.random.uniform(y_min, y_max)
        self.o = np.random.uniform(0, 360)
        self.v = np.random.uniform(0, self.v_max)
        self.dv = np.random.uniform(-self.dv_max, self.dv_max) # NOT NEEDED?


