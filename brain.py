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


class Generation:
    def __init__(self, gen_id=None, move_memory=4, filename="save_data.json"):

        self.size = 0
        self._agent_ids = 0

        self.all_agents = {}

        self.filename = filename
        if gen_id is not None:
            self.gen_id = gen_id
        else:
            self.gen_id = self.pull_generation_count()

    def __getitem__(self, key):
        return self.all_agents[key]

    def add_agents(self, agents):
        for agent in agents:
            self.add_agent(agent)

    def add_agent(self, agent):
        self.size += 1
        self._agent_ids += 1
        self.all_agents[self._agent_ids] = agent
        agent.id = self._agent_ids

    def remove_agents(self, agents):
        for agent in agents:
            self.remove_agent(agent)

    def remove_agent(self, agent):
        self.size -= 1
        if agent.id not in self.all_agents:
            return
        else:
            id = agent.id
            del self.all_agents[id]
            agent.id = None

    def update_fitness(self, agent_index, reward):
        self.all_agents[agent_index].fitness += reward

    def pull_generation_count(self):
        try:
            with open(self.filename, "rb") as f:
                items = ijson.items(f, "generation_count")
                gen_count = next(items)
        except (FileNotFoundError, StopIteration, ijson.JSONDecodeError):
            gen_count = 0
        return gen_count

    def save_generation_count(self):
        try:
            with open(self.filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        history['generation_count'] = self.gen_id

        with open(self.filename, 'w') as f:
            json.dump(history, f, indent=4)

    def save(self, agent_ids):
        try:
            with open(self.filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        if "agent_data" not in history:
            history["agent_data"] = {}
        if str(self.gen_id) not in history["agent_data"]:
            history["agent_data"][str(self.gen_id)] = {}
        for agent_id in agent_ids:
            history["agent_data"][str(self.gen_id)][str(
                agent_id)] = self.all_agents[agent_id].to_dict()

        with open(self.filename, 'w') as f:
            json.dump(history, f, indent=4)

    def load(self, ids):  # the ids is a list of tuples (generation, id)
        agents = []
        try:
            with open(self.filename, 'rb') as f:
                items = ijson.items(f, "agent_data")
                agent_data = next(items)
        except (FileNotFoundError, StopIteration, ijson.JSONDecodeError):
            return None
        for id in ids:
            agents.append(agent_data[id[0]][id[1]])
        return agents


class Genome:
    def __init__(self, genome_id, weights):
        self.id = genome_id
        self.weights = weights

        self.connections = {}
        self.nodes = {}

    def to_dict(self):
        return {
            "id": self.id,
            # yeah this is here for the future womp womp
            "connections": {str(k): v.to_dict() for k, v in self.connections.items()}
        }


class Connection:
    def __init__(self):
        pass

    def to_dict(self):
        return self.__dict__


class NodeGene:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type


class Agent:
    def __init__(self, genome):
        self.id = None
        self.fitness = None
        self.genome = genome

    def to_dict(self):
        return {
            "fitness": self.fitness,
            "genome_data": self.genome.to_dict()
        }


class Evaluator:
    pass


class Breeder:
    pass


class Simulation:
    def __init__(self):
        self.innovation_tracker = InnovationTracker()
