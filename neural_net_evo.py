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

    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)
        self.x = np.random.uniform(self.x_min, self.x_max)
        self.y = np.random.uniform(self.y_min, self.y_max)
        self.o = np.random.uniform(0, 360)
        self.v = np.random.uniform(0, self.v_max)
        self.dv = np.random.uniform(-self.dv_max, self.dv_max) # NOT NEEDED?
        self.d_food = 9999
        self.o_food = 0
        self.fitness = 0

        # If weights are passed then use those, otherwise generate randomly.
        if self.w_input_hidden is None:
            self.w_input_hidden = np.random.uniform(
                -1, 1, (self.n_hidden_nodes, self.n_input_nodes)
                )
        if self.w_hidden_output is None:
            self.w_hidden_output = np.random.uniform(
                -1, 1, (self.n_output_nodes, self.n_hidden_nodes)
                )


    def think(self):

        # Run MLP.
        af = lambda x: np.tanh(x) # Activation function
        h1 = af(np.dot(self.w_input_hidden, self.o_food)) # Hidden layer 1
        out = af(np.dot(self.w_hidden_output, h1)) # Output layer

        # Update dv and do with MLP response.
        self.nn_dv = float(out[0]) # [-1, 1] (accelerate=1, deaccelerate=-1)
        self.nn_do = float(out[1]) # [-1, 1] (left=1, right=-1)


    def update_o(self, dt):
        
        # Update the organism's heading
        self.o += self.nn_do * self.do_max * dt
        self.o = self.o % 360


    def update_v(self, dt):
        
        # Update the organism's velocity
        self.v += self.nn_dv * self.dv_max * dt
        if self.v < 0:
            self.v = 0
        if self.v > self.v_max:
            self.v = self.v_max

    def update_pos(self, dt):
        dx = self.v * np.cos(np.radians(self.o)) * dt
        dy = self.v * np.sin(np.radians(self.o)) * dt
        self.x += dx
        self.y += dy


class Environment(object):

    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

        self.time = 1
        self.generation = 1
        
        self.foods = [
        Food(self.x_min, self.x_max, self.y_min, self.y_max, self.food_value)
        for i in range(self.num_food)
        ]

        self.organisms = [
            Organism(
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                v_max=self.v_max,
                dv_max=self.dv_max,
                do_max=self.do_max,
                n_input_nodes=self.n_input_nodes,
                n_hidden_nodes=self.n_hidden_nodes,
                n_output_nodes=self.n_output_nodes,
                tolerance=self.tolerance,
                w_input_hidden=None,
                w_hidden_output=None,
                name=f'gen[{self.generation}]-org[{i + 1}]'
                ) for i in range(self.num_orgs)
        ]


    def run(self):
        total_time_steps = int(self.gen_time / self.dt)
        for gen in range(self.num_gens):
            print(f'Simulating generation {gen}')        
            for step in range(total_time_steps):
                self.tick()
            self.evolve()



    def tick(self):

        # Update fitness.
        for food in self.foods:
            for org in self.organisms:
                food_org_dist = self._dist(org.x, org.y, food.x, food.y)
                if food_org_dist <= self.tolerance:
                    org.fitness += food.value
                    food.respawn(
                        self.x_min,
                        self.x_max,
                        self.y_min,
                        self.y_max,
                        self.food_value
                        )
                org.d_food = 9999
                org.o_food = 0
        
        # Calculate heading to nearest food.
        for food in self.foods:
            for org in self.organisms:
                food_org_dist = self._dist(org.x, org.y, food.x, food.y)
                
                # Determine if this is the closest food.
                if food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    org.o_food = self._calc_heading(org, food)

        # Get organism response.
        for org in self.organisms:
            org.think()

        # Update each organism's position and velocity.
        for org in self.organisms:
            org.update_o(self.dt)
            org.update_v(self.dt)
            org.update_pos(self.dt)

    
    def evolve(self):

        keep_num = int(np.floor(self.elitism * self.num_orgs))
        new_num = self.num_orgs - keep_num

        # TODO: ADD STATISTICS REPORTING

        # Elitism: keep the best performing organisms.
        orgs_sorted = sorted(
            self.organisms, key=lambda x: x.fitness, reverse=True
            )
        orgs_new = [
            Organism(
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                v_max=self.v_max,
                dv_max=self.dv_max,
                do_max=self.do_max,
                n_input_nodes=self.n_input_nodes,
                n_hidden_nodes=self.n_hidden_nodes,
                n_output_nodes=self.n_output_nodes,
                tolerance=self.tolerance,
                w_input_hidden=orgs_sorted[i].w_input_hidden,
                w_hidden_output=orgs_sorted[i].w_hidden_output,
                name=orgs_sorted[i].name
                ) for i in range(keep_num)
        ]

        # Generate new organisms.
        for i in range(new_num):

            # Perform truncation selection.
            org_1, org_2 = np.random.choice(orgs_sorted, 2)

            # Perform crossover.
            crossover_weight = np.random.uniform(0, 1)
            w_input_hidden_new = (
                (crossover_weight * org_1.w_input_hidden)
                + ((1 - crossover_weight) * org_2.w_input_hidden)
            )
            w_hidden_output_new = (
                (crossover_weight * org_1.w_hidden_output)
                + ((1 - crossover_weight) * org_2.w_hidden_output)
            )

            




    def _dist(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def _calc_heading(self, org, food):
        d_x = food.x - org.x
        d_y = food.y - org.y
        theta_d = np.degrees(np.arctan2(d_y, d_x)) - org.o
        if abs(theta_d) > 180: theta_d += 360
        return theta_d / 180











        








