import json
import numpy as np


class Population:
    def __init__(self, size, input_size=3, output_size=3):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size

        self.all_weights = np.random.randn(size, input_size, output_size)

        self.fitness_scores = np.zeros(size)

        self.gen_id = self.update_population_count()

    def get_outputs(self, inputs):
        """
        Calculates the choices for the ENTIRE population at once.
        inputs: A 1D array of the last moves [r, p, s]
        """
        raw_outputs = np.matmul(inputs, self.all_weights)

        # Return the index of the highest score for every agent
        return np.argmax(raw_outputs, axis=1)

    def update_fitness(self, agent_index, reward):
        self.fitness_scores[agent_index] += reward

    def update_population_count(self, filename="save_data.json"):
        try:
            with open(filename, 'r') as f:
                # FIX: changed 'text' to 'history' to match the rest of the function
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        save_key = "population_count"
        if save_key in history:
            history[save_key] += 1
        else:
            history[save_key] = 1

        pop_count = history[save_key]
        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)
        return pop_count

    def save_agent_from_pop(self, agent_id, filename="save_data.json"):
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        # The save_key remains the same for your history tracking
        save_key = f"gen_{self.gen_id}_id_{agent_id}"

        # We pull the weights directly from the Mega-Tensor at the agent's index
        history[save_key] = {
            'id': agent_id,
            'generation_id': self.gen_id,
            'weights': self.all_weights[agent_id].tolist(),
            # Ensure it's a JSON-friendly float
            'fitness': float(self.fitness_scores[agent_id])
        }

        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)

    def load_agent_into_pop(self, agent_id, gen_id, filename="save_data.json"):
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("No saved data found.")
            return False

        save_key = f"gen_{gen_id}_id_{agent_id}"
        data = history.get(save_key)

        if data:
            # We inject the loaded weights directly into the Mega-Tensor
            self.all_weights[agent_id] = np.array(data['weights'])
            self.fitness_scores[agent_id] = data['fitness']
            print(f"Agent {agent_id} loaded from Gen {gen_id}.")
            return True

        print("Agent not found in history.")
        return False
