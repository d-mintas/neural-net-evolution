from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import radians, degrees, sqrt, atan2


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
        self.dv = np.random.uniform(-self.dv_max, self.dv_max)
        self.d_food = 9999
        self.o_food = 0
        self.fitness = 0
        
        # Create offspring between its two parents.
        if 'x_inherited' in self.__dict__.keys():
            self.x = self.x_inherited
        if 'y_inherited' in self.__dict__.keys():
            self.y = self.y_inherited

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
        dx = self.v * np.cos(radians(self.o)) * dt
        dy = self.v * np.sin(radians(self.o)) * dt
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
                mutation_rate=self.mutation_rate,
                w_input_hidden=None,
                w_hidden_output=None,
                color=self.colors[np.random.randint(0, len(self.colors))],
                name=f'gen[{self.generation}]-org[{i + 1}]'
                ) for i in range(self.num_orgs)
        ]


    def run(self):
        
        total_time_steps = int(self.gen_time / self.dt)
        
        for gen in range(self.num_gens):
            print(f'Simulating...')
            for step in tqdm(range(total_time_steps)):
                yield {
                'foods': [(food.x, food.y) for food in self.foods],
                'orgs': [
                (org.x, org.y, org.fitness, org.color)
                for org in self.organisms
                ]
                }
                self.tick()
            fitnesses = sorted([org.fitness for org in self.organisms])
            best = fitnesses[-1]
            worst = fitnesses[0]
            average = np.mean(fitnesses)
            
            print(
                f'\nGEN {gen + 1} | \
                Best: {best} Average: {average} Worst: {worst}'
                )

            self.evolve()


    def tick(self):

        # Build distances matrices.
        self.matrices = []
        for food in self.foods:
            fa = np.array([food.x, food.y])
            oa = np.array([(org.x, org.y) for org in self.organisms])
            distances = (fa - oa) ** 2
            distances = np.sum(distances, axis=1)
            distances = np.sqrt(distances)
            self.matrices.append(distances)

        # Update fitness.
        for i, food in enumerate(self.foods):
            for j, org in enumerate(self.organisms):
                food_org_dist = self.matrices[i][j]
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
        for i, food in enumerate(self.foods):
            for j, org in enumerate(self.organisms):
                food_org_dist = self.matrices[i][j]
                # food_org_dist = self._dist(org.x, org.y, food.x, food.y)
                # food_org_dist = sqrt((food.x - org.x)**2 + (food.y - org.y)**2)
                
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

        elite_num = int(np.floor(self.elitism * self.num_orgs))

        # Elitism: keep the best performing organisms.
        orgs_sorted = sorted(
            self.organisms, key=lambda x: x.fitness, reverse=True
            )

        orgs_elite = orgs_sorted[0:elite_num]
        orgs_new = []
        # orgs_new = [
        #     Organism(
        #         x_min=self.x_min,
        #         x_max=self.x_max,
        #         y_min=self.y_min,
        #         y_max=self.y_max,
        #         v_max=self.v_max,
        #         dv_max=self.dv_max,
        #         do_max=self.do_max,
        #         n_input_nodes=self.n_input_nodes,
        #         n_hidden_nodes=self.n_hidden_nodes,
        #         n_output_nodes=self.n_output_nodes,
        #         tolerance=self.tolerance,
        #         mutation_rate=self.mutation_rate,
        #         w_input_hidden=orgs_sorted[i].w_input_hidden,
        #         w_hidden_output=orgs_sorted[i].w_hidden_output,
        #         color=orgs_sorted[i].color,
        #         name=orgs_sorted[i].name
        #         ) for i in range(keep_num)
        # ]

        # Generate new organisms.
        for i in range(self.num_orgs):

            # Perform truncation selection.
            match = False
            while not match:
                org_1, org_2 = np.random.choice(orgs_elite, 2)
                match = (
                    sqrt(
                        (org_1.color[0] - org_2.color[0]) ** 2
                        + (org_1.color[1] - org_2.color[1]) ** 2
                        + (org_1.color[2] - org_2.color[2]) ** 2
                        ) <= 400
                    )

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

            # Perform mutation.
            if np.random.uniform(0, 1) <= self.mutation_rate:
                
                if np.random.randint(0, 1) == 0:
                    # Mutate [input -> hidden] weights.
                    w_row = np.random.randint(0, self.n_hidden_nodes - 1)
                    w_input_hidden_new[w_row] *= np.random.uniform(.9, 1.1)
                    if w_input_hidden_new[w_row] > 1:
                        w_input_hidden_new[w_row] = 1
                    elif w_input_hidden_new[w_row] < -1:
                        w_input_hidden_new[w_row] = -1
                
                else:
                    # Mutate [hidden -> output] weights.
                    w_row = np.random.randint(0, self.n_output_nodes - 1)
                    w_col = np.random.randint(0, self.n_hidden_nodes - 1)
                    w_hidden_output_new[w_row][w_col] *= np.random.uniform(
                        .9, 1.1
                        )
                    if w_hidden_output_new[w_row][w_col] > 1:
                        w_hidden_output_new[w_row][w_col] = 1
                    if w_hidden_output_new[w_row][w_col] < -1:
                        w_hidden_output_new[w_row][w_col] = - 1

            orgs_new.append(
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
                mutation_rate=self.mutation_rate,
                w_input_hidden=w_input_hidden_new,
                w_hidden_output=w_hidden_output_new,
                name=f'gen[{self.generation}]-org[{i + 1}]',
                x_inherited=np.mean([org_1.x, org_2.x]),
                y_inherited=np.mean([org_1.y, org_2.y]),
                color=[
                round(e) for e in np.mean([org_1.color, org_2.color], axis=0)
                ]
                    )
                )
        self.organisms = orgs_new


    # def evolve(self):

    #     keep_num = int(np.floor(self.elitism * self.num_orgs))
    #     new_num = self.num_orgs - keep_num

    #     # Elitism: keep the best performing organisms.
    #     orgs_sorted = sorted(
    #         self.organisms, key=lambda x: x.fitness, reverse=True
    #         )
    #     orgs_new = [
    #         Organism(
    #             x_min=self.x_min,
    #             x_max=self.x_max,
    #             y_min=self.y_min,
    #             y_max=self.y_max,
    #             v_max=self.v_max,
    #             dv_max=self.dv_max,
    #             do_max=self.do_max,
    #             n_input_nodes=self.n_input_nodes,
    #             n_hidden_nodes=self.n_hidden_nodes,
    #             n_output_nodes=self.n_output_nodes,
    #             tolerance=self.tolerance,
    #             mutation_rate=self.mutation_rate,
    #             w_input_hidden=orgs_sorted[i].w_input_hidden,
    #             w_hidden_output=orgs_sorted[i].w_hidden_output,
    #             color=orgs_sorted[i].color,
    #             name=orgs_sorted[i].name
    #             ) for i in range(keep_num)
    #     ]

    #     # Generate new organisms.
    #     for i in range(new_num):

    #         # Perform truncation selection.
    #         org_1, org_2 = np.random.choice(orgs_sorted, 2)

    #         # Perform crossover.
    #         crossover_weight = np.random.uniform(0, 1)
    #         w_input_hidden_new = (
    #             (crossover_weight * org_1.w_input_hidden)
    #             + ((1 - crossover_weight) * org_2.w_input_hidden)
    #         )
    #         w_hidden_output_new = (
    #             (crossover_weight * org_1.w_hidden_output)
    #             + ((1 - crossover_weight) * org_2.w_hidden_output)
    #         )

    #         # Perform mutation.
    #         if np.random.uniform(0, 1) <= self.mutation_rate:
                
    #             if np.random.randint(0, 1) == 0:
    #                 # Mutate [input -> hidden] weights.
    #                 w_row = np.random.randint(0, self.n_hidden_nodes - 1)
    #                 w_input_hidden_new[w_row] *= np.random.uniform(.9, 1.1)
    #                 if w_input_hidden_new[w_row] > 1:
    #                     w_input_hidden_new[w_row] = 1
    #                 elif w_input_hidden_new[w_row] < -1:
    #                     w_input_hidden_new[w_row] = -1
                
    #             else:
    #                 # Mutate [hidden -> output] weights.
    #                 w_row = np.random.randint(0, self.n_output_nodes - 1)
    #                 w_col = np.random.randint(0, self.n_hidden_nodes - 1)
    #                 w_hidden_output_new[w_row][w_col] *= np.random.uniform(
    #                     .9, 1.1
    #                     )
    #                 if w_hidden_output_new[w_row][w_col] > 1:
    #                     w_hidden_output_new[w_row][w_col] = 1
    #                 if w_hidden_output_new[w_row][w_col] < -1:
    #                     w_hidden_output_new[w_row][w_col] = - 1

    #         orgs_new.append(
    #             Organism(
    #             x_min=self.x_min,
    #             x_max=self.x_max,
    #             y_min=self.y_min,
    #             y_max=self.y_max,
    #             v_max=self.v_max,
    #             dv_max=self.dv_max,
    #             do_max=self.do_max,
    #             n_input_nodes=self.n_input_nodes,
    #             n_hidden_nodes=self.n_hidden_nodes,
    #             n_output_nodes=self.n_output_nodes,
    #             tolerance=self.tolerance,
    #             mutation_rate=self.mutation_rate,
    #             w_input_hidden=w_input_hidden_new,
    #             w_hidden_output=w_hidden_output_new,
    #             name=f'gen[{self.generation}]-org[{i + 1}]',
    #             x_inherited=np.mean([org_1.x, org_2.x]),
    #             y_inherited=np.mean([org_1.y, org_2.y]),
    #             color=[
    #             round(e) for e in np.mean([org_1.color, org_2.color], axis=0)
    #             ]
    #                 )
    #             )
    #     self.organisms = orgs_new


    def _calc_heading(self, org, food):
        d_x = food.x - org.x
        d_y = food.y - org.y
        theta_d = degrees(atan2(d_y, d_x)) - org.o
        if abs(theta_d) > 180: theta_d += 360
        return theta_d / 180











        








