import ijson
import json
import numpy as np


class InnovationTracker:
    def __init__(self, filename="save_data.json"):
        self.filename = filename
        initial_data = self.__pull_innovation_data()

        if initial_data:
            self.innovations = initial_data.get('innovation_dict', {})
            self.current_innovation_number = initial_data.get(
                'innovation_number', 0)
        else:
            self.innovations = {}
            self.current_innovation_number = 0

    def get_innovation_number(self):
        self.current_innovation_number += 1
        return self.current_innovation_number

    def key_to_string(self, key):
        # Turns (1, 2) into "1-2"
        return "-".join(map(str, key))

    def string_to_key(self, key):
        # Turns "1-2" back into (1, 2)
        return tuple(map(int, key.split("-")))

    def add_innovation(self, key):
        if key not in self.innovations:
            num = self.get_innovation_number()
            self.innovations[key] = num
            return num
        return self.innovations[key]

    def save_innovation_data(self):
        try:
            with open(self.filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        history['innovations'] = {
            'innovation_number': self.current_innovation_number,
            # Convert tuple keys to string keys
            'innovation_dict': {self.key_to_string(k): v for k, v in self.innovations.items()}
        }

        with open(self.filename, 'w') as f:
            json.dump(history, f, indent=4)

    def __pull_innovation_data(self):
        try:
            with open(self.filename, 'rb') as f:
                items = ijson.items(f, 'innovations')
                innovation_data = next(items)
        except (FileNotFoundError, StopIteration, ijson.JSONDecodeError):
            return None
        # convert the string keys back to tuple keys
        innovation_data['innovation_dict'] = {self.string_to_key(
            k): v for k, v in innovation_data['innovation_dict'].items()}
        return innovation_data


class Population:
    def __init__(self, size, input_size=3, output_size=3, filename="save_data.json"):

        self.size = size
        self.input_size = input_size
        self.output_size = output_size

        self.all_agents = {}
        self.all_weights = {}

        self.fitness_scores = {}

        self.gen_id = self.update_population_count()

    def update_fitness(self, agent_index, reward):
        self.fitness_scores[agent_index] += reward

    # TODO: add a method to save an agent into the json, and a method to load an agent from the json, aswell as helpers
