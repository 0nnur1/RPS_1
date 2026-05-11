import copy
import json
import os
import random
import numpy as np

# ---------------- CONFIG ----------------

MEMORY_SIZE = 5
OUTPUT_SIZE = 3

HISTORY_SIZE = MEMORY_SIZE * 2 * 3
INPUT_SIZE = HISTORY_SIZE + 1  # +1 bias

# Lower tick count reduces activation snowballing and output saturation.
TICKS_PER_MOVE = 3
ROUNDS_PER_MATCH = 20
POPULATION_SIZE = 100
MATCHES_PER_AGENT = 6

MUTATION_WEIGHT_CHANCE = 0.8
MUTATION_STRUCTURAL_CHANCE = 0.03
ADD_CONNECTION_CHANCE = 0.08

HISTORY_DECAY = 0.90
CONFIDENCE_PENALTY_SCALE = 0.02
ENTROPY_BONUS_SCALE = 0.75
REPEAT_MOVE_PENALTY = 0.10

RANDOM_BASELINE_MATCHES = 1
FIXED_EVAL_POOL_SIZE = 10
ELITE_COUNT = 5
STATE_FILE = "save_data.json"


# ---------------- HELPERS ----------------

def activate(x):
    """
    Tanh keeps activations centered around 0 and avoids sigmoid's one-way drift.
    """
    x = np.clip(x, -20, 20)
    return np.tanh(x)


def softmax(values, temperature=1.0):
    vals = np.array(values, dtype=float)
    temperature = max(1e-6, float(temperature))
    vals = vals / temperature
    vals = vals - np.max(vals)
    exps = np.exp(vals)
    total = np.sum(exps)
    if total <= 0 or not np.isfinite(total):
        return np.ones(len(values), dtype=float) / max(1, len(values))
    return exps / total


def confidence_collapse(outputs):
    if len(outputs) < 2:
        return 0.0
    vals = sorted(outputs, reverse=True)
    return max(0.0, vals[0] - vals[1])


def normalized_entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0

    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0

    h = -sum(p * np.log(p) for p in probs)
    return float(h / np.log(len(counts)))


# ---------------- GENES ----------------

class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type


class Connection:
    def __init__(self, in_node, out_node, weight, innovation_id, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation_id = innovation_id
        self.enabled = enabled

    def copy(self):
        return Connection(
            self.in_node,
            self.out_node,
            self.weight,
            self.innovation_id,
            self.enabled,
        )

    def to_dict(self):
        return {
            "in_node": self.in_node,
            "out_node": self.out_node,
            "weight": float(self.weight),
            "innovation_id": self.innovation_id,
            "enabled": self.enabled,
        }


# ---------------- INNOVATION TRACKER ----------------

class InnovationTracker:
    def __init__(self):
        self.nodes = {}
        self.innovations = {}
        self.current_node_number = 0
        self.current_innovation_number = 0

    def add_node(self, key):
        key = str(key)
        if key not in self.nodes:
            self.current_node_number += 1
            self.nodes[key] = self.current_node_number
        return self.nodes[key]

    def add_innovation(self, key):
        key = str(key)
        if key not in self.innovations:
            self.current_innovation_number += 1
            self.innovations[key] = self.current_innovation_number
        return self.innovations[key]


# ---------------- GENOME ----------------

class Genome:
    def __init__(self, tracker, initialize=True):
        self.tracker = tracker
        self.nodes = {}
        self.connections = {}
        self.sensor_ids = []
        self.output_ids = []

        if initialize:
            for i in range(INPUT_SIZE):
                nid = self.tracker.add_node(("input", i))
                self.sensor_ids.append(nid)
                self.nodes[nid] = NodeGene(nid, "sensor")

            for i in range(OUTPUT_SIZE):
                nid = self.tracker.add_node(("output", i))
                self.output_ids.append(nid)
                self.nodes[nid] = NodeGene(nid, "output")

            for sid in self.sensor_ids:
                for oid in self.output_ids:
                    inv = self.tracker.add_innovation((sid, oid))
                    self.connections[inv] = Connection(
                        sid,
                        oid,
                        np.random.uniform(-1, 1),
                        inv,
                    )

    def copy(self):
        g = Genome(self.tracker, initialize=False)
        g.sensor_ids = self.sensor_ids[:]
        g.output_ids = self.output_ids[:]
        g.nodes = {
            nid: NodeGene(node.id, node.type)
            for nid, node in self.nodes.items()
        }
        g.connections = {
            inv: conn.copy()
            for inv, conn in self.connections.items()
        }
        return g

    def __deepcopy__(self, memo):
        return self.copy()

    def mutate(self):
        self.mutate_weights()

        if random.random() < MUTATION_STRUCTURAL_CHANCE:
            self.mutate_add_node()

        if random.random() < ADD_CONNECTION_CHANCE:
            self.mutate_add_connection()

    def mutate_weights(self):
        for conn in self.connections.values():
            r = random.random()
            if r < 0.1:
                conn.weight = np.random.uniform(-1, 1)
            elif r < 0.9:
                conn.weight += np.random.normal(0, 0.15)
                conn.weight = np.clip(conn.weight, -3, 3)

    def mutate_add_node(self):
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return

        conn = random.choice(enabled)
        conn.enabled = False

        new_node_id = self.tracker.add_node(("hidden", conn.innovation_id))
        if new_node_id not in self.nodes:
            self.nodes[new_node_id] = NodeGene(new_node_id, "hidden")

        inv1 = self.tracker.add_innovation((conn.in_node, new_node_id))
        inv2 = self.tracker.add_innovation((new_node_id, conn.out_node))

        self.connections[inv1] = Connection(
            conn.in_node, new_node_id, 1.0, inv1
        )
        self.connections[inv2] = Connection(
            new_node_id, conn.out_node, conn.weight, inv2
        )

    def mutate_add_connection(self):
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return

        for _ in range(50):
            a = random.choice(node_ids)
            b = random.choice(node_ids)

            if a == b:
                continue

            a_type = self.nodes[a].type
            b_type = self.nodes[b].type

            if b_type == "sensor":
                continue

            if a_type == "output" and b_type == "output":
                continue

            exists = False
            for c in self.connections.values():
                if c.in_node == a and c.out_node == b:
                    exists = True
                    break

            if exists:
                continue

            inv = self.tracker.add_innovation((a, b))
            self.connections[inv] = Connection(
                a,
                b,
                np.random.uniform(-1, 1),
                inv,
            )
            return


# ---------------- AGENT ----------------

class Agent:
    def __init__(self, genome, agent_id):
        self.genome = genome.copy()
        self.id = agent_id
        self.fitness = 0.0
        self.last_outputs = [0.0] * OUTPUT_SIZE

        self.runtime_connections = {}
        self.node_values = {
            nid: {"sum": 0.0, "output": 0.0}
            for nid in self.genome.nodes
        }
        self.active_nodes = set()
        self.assemble_network()

    def assemble_network(self):
        self.runtime_connections = {nid: [] for nid in self.genome.nodes}
        for conn in self.genome.connections.values():
            if conn.enabled:
                self.runtime_connections[conn.in_node].append(
                    (conn.out_node, conn.weight)
                )

    def reset_brain(self):
        for nid in self.node_values:
            self.node_values[nid]["sum"] = 0.0
            self.node_values[nid]["output"] = 0.0
        self.active_nodes = set()

    def walk_tick(self):
        next_active = set()

        for nid in self.active_nodes:
            output = float(self.node_values[nid]["output"])
            for target, weight in self.runtime_connections.get(nid, []):
                self.node_values[target]["sum"] += output * weight
                next_active.add(target)

        # Clear stale outputs for non-sensor nodes that were not activated.
        # This prevents one lucky move from sticking forever.
        for nid in self.node_values:
            if nid not in next_active and self.genome.nodes[nid].type != "sensor":
                self.node_values[nid]["output"] = 0.0

        for nid in next_active:
            x = self.node_values[nid]["sum"]
            self.node_values[nid]["output"] = activate(x)
            self.node_values[nid]["sum"] = 0.0

        self.active_nodes = next_active

    def decide(self):
        for _ in range(TICKS_PER_MOVE):
            self.walk_tick()

        outputs = [
            float(self.node_values[oid]["output"])
            for oid in self.genome.output_ids
        ]
        self.last_outputs = outputs[:]

        probs = softmax(outputs, temperature=1.0)
        return int(np.random.choice(len(outputs), p=probs))


class RandomOpponent:
    def __init__(self, genome_template):
        self.genome = genome_template.copy()
        self.id = -1
        self.fitness = 0.0
        self.last_outputs = [1 / 3, 1 / 3, 1 / 3]

        self.node_values = {
            nid: {"sum": 0.0, "output": 0.0}
            for nid in self.genome.nodes
        }
        self.active_nodes = set()

    def reset_brain(self):
        for nid in self.node_values:
            self.node_values[nid]["sum"] = 0.0
            self.node_values[nid]["output"] = 0.0
        self.active_nodes = set()

    def walk_tick(self):
        return set()

    def decide(self):
        choice = random.randint(0, OUTPUT_SIZE - 1)
        self.last_outputs = [0.0] * OUTPUT_SIZE
        self.last_outputs[choice] = 1.0
        return choice


# ---------------- SIMULATION ----------------

class Simulation:
    def __init__(self):
        self.tracker = InnovationTracker()
        self.generation = 0
        self.agents = []
        self.eval_pool = []

        if not self.load_state():
            self.agents = [
                Agent(Genome(self.tracker), i)
                for i in range(POPULATION_SIZE)
            ]
            self.refresh_eval_pool()

    def refresh_eval_pool(self):
        if not self.agents:
            self.eval_pool = []
            return
        self.eval_pool = random.sample(
            self.agents,
            k=min(FIXED_EVAL_POOL_SIZE, len(self.agents))
        )

    def run_match(self, a1, a2):
        """
        Returns fitness deltas for both participants.
        This avoids mutating shared agent state during evaluation.
        """
        a1.reset_brain()
        a2.reset_brain()

        h1 = [0.0] * HISTORY_SIZE
        h2 = [0.0] * HISTORY_SIZE

        move_counts_1 = [0, 0, 0]
        move_counts_2 = [0, 0, 0]

        last_move_1 = None
        last_move_2 = None
        repeat_streak_1 = 0
        repeat_streak_2 = 0

        fitness_1 = 0.0
        fitness_2 = 0.0

        for _ in range(ROUNDS_PER_MATCH):
            h1 = [x * HISTORY_DECAY for x in h1]
            h2 = [x * HISTORY_DECAY for x in h2]

            for i, val in enumerate(h1):
                sid = a1.genome.sensor_ids[i]
                a1.node_values[sid]["output"] = val

            for i, val in enumerate(h2):
                sid = a2.genome.sensor_ids[i]
                a2.node_values[sid]["output"] = val

            bias_index = INPUT_SIZE - 1
            a1.node_values[a1.genome.sensor_ids[bias_index]]["output"] = 1.0
            a2.node_values[a2.genome.sensor_ids[bias_index]]["output"] = 1.0

            a1.active_nodes = set(a1.genome.sensor_ids)
            a2.active_nodes = set(a2.genome.sensor_ids)

            m1 = a1.decide()
            m2 = a2.decide()

            move_counts_1[m1] += 1
            move_counts_2[m2] += 1

            if m1 == last_move_1:
                repeat_streak_1 += 1
            else:
                repeat_streak_1 = 1
            last_move_1 = m1

            if m2 == last_move_2:
                repeat_streak_2 += 1
            else:
                repeat_streak_2 = 1
            last_move_2 = m2

            if m1 == m2:
                fitness_1 += 0.1
                fitness_2 += 0.1
            elif (m1 - m2) % 3 == 1:
                fitness_1 += 1.0
                fitness_2 -= 1.0
            else:
                fitness_2 += 1.0
                fitness_1 -= 1.0

            fitness_1 -= confidence_collapse(a1.last_outputs) * \
                CONFIDENCE_PENALTY_SCALE
            fitness_2 -= confidence_collapse(a2.last_outputs) * \
                CONFIDENCE_PENALTY_SCALE

            if repeat_streak_1 > 2:
                fitness_1 -= REPEAT_MOVE_PENALTY
            if repeat_streak_2 > 2:
                fitness_2 -= REPEAT_MOVE_PENALTY

            new1 = [0.0] * 6
            new2 = [0.0] * 6

            new1[m2] = 1.0
            new1[3 + m1] = 1.0

            new2[m1] = 1.0
            new2[3 + m2] = 1.0

            h1 = h1[6:] + new1
            h2 = h2[6:] + new2

        fitness_1 += ENTROPY_BONUS_SCALE * normalized_entropy(move_counts_1)
        fitness_2 += ENTROPY_BONUS_SCALE * normalized_entropy(move_counts_2)

        return fitness_1, fitness_2

    def evaluate(self):
        # Evaluate everyone fresh.
        fitness_deltas = {a: 0.0 for a in self.agents}

        for agent in self.agents:
            fixed_opps = [opp for opp in self.eval_pool if opp is not agent]
            random.shuffle(fixed_opps)

            chosen = fixed_opps[: min(2, len(fixed_opps))]
            remaining = MATCHES_PER_AGENT - len(chosen)

            random_pool = [
                a for a in self.agents
                if a is not agent and a not in chosen
            ]

            if remaining > 0 and random_pool:
                chosen.extend(
                    random.sample(
                        random_pool,
                        k=min(remaining, len(random_pool))
                    )
                )

            for opp in chosen:
                d1, d2 = self.run_match(agent, opp)
                fitness_deltas[agent] += d1
                fitness_deltas[opp] += d2

            for _ in range(RANDOM_BASELINE_MATCHES):
                d1, _ = self.run_match(agent, RandomOpponent(agent.genome))
                fitness_deltas[agent] += d1

        for a in self.agents:
            fitness_deltas[a] -= len(a.genome.connections) * 0.001

        for a in self.agents:
            a.fitness = fitness_deltas[a]

    def crossover(self, parents):
        child = Genome(self.tracker, initialize=False)

        for p in parents:
            for nid, node in p.genome.nodes.items():
                if nid not in child.nodes:
                    child.nodes[nid] = NodeGene(nid, node.type)

        for nid in parents[0].genome.sensor_ids:
            if nid not in child.nodes:
                child.nodes[nid] = NodeGene(nid, "sensor")
        for nid in parents[0].genome.output_ids:
            if nid not in child.nodes:
                child.nodes[nid] = NodeGene(nid, "output")

        all_innovations = set()
        for p in parents:
            all_innovations.update(p.genome.connections.keys())

        for inv in all_innovations:
            owners = [p for p in parents if inv in p.genome.connections]
            chosen = random.choice(owners)
            gene = chosen.genome.connections[inv].copy()

            if any(not p.genome.connections[inv].enabled for p in owners):
                if random.random() < 0.75:
                    gene.enabled = False

            child.connections[inv] = gene

        child.sensor_ids = parents[0].genome.sensor_ids[:]
        child.output_ids = parents[0].genome.output_ids[:]

        return child

    def evolve(self):
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        print(
            f"Generation {self.generation} | Best: {self.agents[0].fitness:.2f}")

        survivors = self.agents[: max(2, POPULATION_SIZE // 2)]
        next_gen = []

        for i, elite in enumerate(survivors[:ELITE_COUNT]):
            next_gen.append(Agent(elite.genome, i))

        while len(next_gen) < POPULATION_SIZE:
            pool = survivors[: min(20, len(survivors))]
            if len(pool) >= 3:
                parents = random.sample(pool, 3)
            else:
                parents = [random.choice(pool)] * 3

            child = self.crossover(parents)
            child.mutate()
            next_gen.append(Agent(child, len(next_gen)))

        self.agents = next_gen
        self.refresh_eval_pool()
        self.generation += 1

        if self.generation % 10 == 0:
            self.save_state()

    def train(self, generations):
        for _ in range(generations):
            self.evaluate()
            self.evolve()

    # ---------------- SAVE / LOAD ----------------

    def save_state(self, path=STATE_FILE):
        data = {
            "generation": self.generation,
            "tracker": {
                "nodes": self.tracker.nodes,
                "innovations": self.tracker.innovations,
                "current_node_number": self.tracker.current_node_number,
                "current_innovation_number": self.tracker.current_innovation_number,
            },
            "eval_pool_ids": [a.id for a in self.eval_pool],
            "agents": []
        }

        for a in self.agents:
            data["agents"].append({
                "id": a.id,
                "fitness": a.fitness,
                "sensor_ids": a.genome.sensor_ids,
                "output_ids": a.genome.output_ids,
                "nodes": {
                    str(nid): node.type
                    for nid, node in a.genome.nodes.items()
                },
                "connections": [
                    c.to_dict()
                    for c in a.genome.connections.values()
                ],
            })

        with open(path, "w") as f:
            json.dump(data, f)

    def load_state(self, path=STATE_FILE):
        if not os.path.exists(path):
            return False

        with open(path, "r") as f:
            data = json.load(f)

        self.generation = data.get("generation", 0)

        tracker_data = data.get("tracker", {})
        self.tracker.nodes = tracker_data.get("nodes", {})
        self.tracker.innovations = tracker_data.get("innovations", {})
        self.tracker.current_node_number = tracker_data.get(
            "current_node_number", 0)
        self.tracker.current_innovation_number = tracker_data.get(
            "current_innovation_number", 0)

        self.agents = []

        for a_data in data.get("agents", []):
            g = Genome(self.tracker, initialize=False)
            g.sensor_ids = a_data.get("sensor_ids", [])
            g.output_ids = a_data.get("output_ids", [])

            g.nodes = {
                int(nid): NodeGene(int(nid), ntype)
                for nid, ntype in a_data.get("nodes", {}).items()
            }

            g.connections = {}
            for conn_data in a_data.get("connections", []):
                conn = Connection(
                    conn_data["in_node"],
                    conn_data["out_node"],
                    float(conn_data["weight"]),
                    conn_data["innovation_id"],
                    conn_data.get("enabled", True),
                )
                g.connections[conn.innovation_id] = conn

            agent = Agent(g, a_data.get("id", 0))
            agent.fitness = a_data.get("fitness", 0.0)
            self.agents.append(agent)

        id_map = {a.id: a for a in self.agents}
        eval_ids = data.get("eval_pool_ids", [])
        self.eval_pool = [id_map[i] for i in eval_ids if i in id_map]

        if not self.eval_pool and self.agents:
            self.refresh_eval_pool()

        return True


# ---------------- LIVE PLAY ----------------

MOVES = {
    0: "Rock ✊",
    1: "Paper ✋",
    2: "Scissors ✌️",
}

INPUT_MAP = {
    "r": 0,
    "p": 1,
    "s": 2,
}


def play_vs_ai():
    sim = Simulation()

    if not sim.agents:
        print("No agents loaded.")
        return

    print("\n--- AI SELECTION ---")
    print("[1] Highest Fitness Agent")
    print("[2] Most Recent Agent")

    choice = input("Selection > ").strip()

    if choice == "1":
        ai = max(sim.agents, key=lambda a: a.fitness)
    else:
        ai = sim.agents[-1]

    print(f"\nLoaded Agent [ID: {ai.id}] [Fitness: {ai.fitness:.2f}]")

    ai.reset_brain()

    h_ai = [0.0] * HISTORY_SIZE
    player_score = 0
    ai_score = 0
    draws = 0
    round_num = 1

    print("\n--- MATCH STARTED ---")
    print("Type: r / p / s")
    print("Ctrl+C to quit.\n")

    try:
        while True:
            user_input = input(f"Round {round_num} > ").strip().lower()

            if user_input not in INPUT_MAP:
                print("Invalid input.")
                continue

            player_move = INPUT_MAP[user_input]

            for i, val in enumerate(h_ai):
                sid = ai.genome.sensor_ids[i]
                ai.node_values[sid]["output"] = val

            bias_index = INPUT_SIZE - 1
            ai.node_values[ai.genome.sensor_ids[bias_index]]["output"] = 1.0

            ai.active_nodes = set(ai.genome.sensor_ids)

            ai_move = ai.decide()

            print(f"\nPlayer: {MOVES[player_move]}")
            print(f"AI:     {MOVES[ai_move]}")

            if player_move == ai_move:
                draws += 1
                print("Result: Draw")
            elif (player_move - ai_move) % 3 == 1:
                player_score += 1
                print("Result: You Win")
            else:
                ai_score += 1
                print("Result: AI Wins")

            print(
                f"\nScore | You: {player_score} AI: {ai_score} Draws: {draws}")

            new_entry = [0.0] * 6
            new_entry[player_move] = 1.0
            new_entry[3 + ai_move] = 1.0
            h_ai = h_ai[6:] + new_entry

            round_num += 1

    except KeyboardInterrupt:
        print("\n\n--- MATCH TERMINATED BY USER ---")
        sim.save_state()


# ---------------- MAIN ----------------

if __name__ == "__main__":
    sim = Simulation()

    try:
        user_input = input(
            "Enter number of generations to train, 'inf' to run forever, or 'p' to play > "
        ).strip().lower()

        if user_input == "p":
            play_vs_ai()

        elif user_input == "inf":
            while True:
                sim.train(1)

        elif user_input.isdigit():
            sim.train(int(user_input))

        else:
            print("Invalid input. Use a number, 'inf', or 'p'.")

    except KeyboardInterrupt:
        print("\nStopping... saving state.")
        sim.save_state()
