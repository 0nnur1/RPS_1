import ijson
import json
import numpy as np
import random  # Fixed: Added missing import for mutate_toggle_connection


class NodeGene:
    """Represents a single 'station' in the brain."""

    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type
        # List of (target_node_id, innovation_id)
        self.outbound_connections = []


class InnovationTracker:
    def __init__(self, filename="save_data.json"):
        self.filename = filename
        initial_data = self.__pull_innovation_data()

        if initial_data:
            self.innovations = initial_data.get('innovation_dict', {})
            self.current_innovation_number = initial_data.get(
                'innovation_number', 0)
            self.nodes = initial_data.get("nodes_dict", {})
            self.current_node_number = initial_data.get(
                "node_number", 0)  # Fixed: Corrected attribute name
        else:
            self.nodes = {}
            self.innovations = {}
            self.current_innovation_number = 0
            self.current_node_number = 0  # Fixed: Corrected attribute name

    def get_innovation_number(self):
        self.current_innovation_number += 1
        return self.current_innovation_number

    def get_node_number(self):
        self.current_node_number += 1
        return self.current_node_number

    def key_to_string(self, key):
        return "-".join(map(str, key))

    def string_to_key(self, key):
        return tuple(map(int, key.split("-")))

    def add_innovation(self, key):
        if key not in self.innovations:
            num = self.get_innovation_number()
            self.innovations[key] = num
            return num
        return self.innovations[key]

    def add_node(self, key):
        if key not in self.nodes:
            num = self.get_node_number()
            self.nodes[key] = num
            return num
        return self.nodes[key]

    def save_innovation_data(self):
        try:
            with open(self.filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        history['innovations'] = {
            'node_number': self.current_node_number,
            'innovation_number': self.current_innovation_number,
            'nodes_dict': {str(k): v for k, v in self.nodes.items()},
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
        innovation_data['innovation_dict'] = {self.string_to_key(
            k): v for k, v in innovation_data['innovation_dict'].items()}
        innovation_data['nodes_dict'] = {
            int(k): v for k, v in innovation_data['nodes_dict'].items()}
        return innovation_data


class Connection:
    """Represents a single 'wire' in the brain, connecting two NodeGenes."""

    def __init__(self, in_node, out_node, weight, innovation_id):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation_id = innovation_id
        self.enabled = True

    def to_dict(self):
        return self.__dict__


class Genome:
    def __init__(self, tracker, memory, input_size, output_size):
        self.input_size = memory * input_size * 2
        self.output_size = output_size
        self.tracker = tracker
        self.connections = {}
        self.nodes = {}  # Now stores NodeGene objects

        # Create Input NodeGenes
        for i in range(self.input_size):
            self.nodes[i] = NodeGene(i, "sensor")

        # Create Output NodeGenes
        for j in range(self.output_size):
            output_id = self.input_size + j
            self.nodes[output_id] = NodeGene(output_id, "output")

        # Initial Full Connection
        for i in range(self.input_size):
            for j in range(self.output_size):
                output_id = self.input_size + j
                innovation_id = self.tracker.add_innovation((i, output_id))
                self.connections[innovation_id] = Connection(
                    i, output_id, np.random.uniform(-1, 1), innovation_id)

    def mutate_add_node(self):
        if not self.connections:
            return
        conn_id = np.random.choice(list(self.connections.keys()))
        old_conn = self.connections[conn_id]
        if not old_conn.enabled:
            return
        old_conn.enabled = False

        new_node_id = self.tracker.add_node(old_conn.innovation_id)
        # Add new NodeGene
        self.nodes[new_node_id] = NodeGene(new_node_id, "hidden")

        id_a = self.tracker.add_innovation((old_conn.in_node, new_node_id))
        self.connections[id_a] = Connection(
            old_conn.in_node, new_node_id, 1.0, id_a)

        id_b = self.tracker.add_innovation((new_node_id, old_conn.out_node))
        self.connections[id_b] = Connection(
            new_node_id, old_conn.out_node, old_conn.weight, id_b)

    def mutate_toggle_connection(self):
        if not self.connections:
            return
        conn_id = random.choice(list(self.connections.keys()))
        self.connections[conn_id].enabled ^= True

    def mutate_weights(self):
        nudge_prob = 0.8
        reset_prob = 0.1
        for conn in self.connections.values():
            rand = np.random.random()
            if rand < reset_prob:
                conn.weight = np.random.uniform(-1, 1)
            elif rand < (reset_prob + nudge_prob):
                nudge = np.random.normal(0, 0.1)  # Center nudges at 0
                conn.weight = np.clip(conn.weight + nudge, -1, 1)

    def to_dict(self):
        return {
            "connections": {str(k): v.to_dict() for k, v in self.connections.items()},
            "nodes": {str(k): v.type for k, v in self.nodes.items()}
        }


class Agent:
    def __init__(self, genome, agent_id):
        self.genome = genome
        self.id = agent_id
        self.fitness = 0

        # Building the 'Live' map from the Genome
        # Each node value now persists until you choose to clear it
        self.node_values = {
            node_id: {"sum": 0.0, "output": 0.0}
            for node_id in self.genome.nodes
        }

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


class Generation:
    def __init__(self, gen_id=None, filename="save_data.json"):

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
