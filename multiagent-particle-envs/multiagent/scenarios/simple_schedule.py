import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.075
        world.agents[0].silent = False
        world.agents[0].channel = 2

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
            agent.goal_c = None
            agent.goal_d = None
            agent.goal_e = None
        goal_landmarks = world.landmarks.copy()
        np.random.shuffle(goal_landmarks)
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = goal_landmarks[0]
        world.agents[0].goal_c = world.agents[2]
        world.agents[0].goal_d = goal_landmarks[1]
        world.agents[0].goal_e = goal_landmarks[2]

        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array(
            [0.45, 0.45, 0.45]
        )
        world.agents[0].goal_c.color = world.agents[0].goal_d.color + np.array(
            [0.45, 0.45, 0.45]
        )
        world.agents[0].color = world.agents[0].goal_e.color + np.array(
            [0.45, 0.45, 0.45]
        )
        world.agents[0].color -= np.array([0.3, 0.3, 0.3])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c * agent.channel)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        rew += self.reward(agent, world)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        a = world.agents[0]
        rew -= np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        rew -= np.sum(np.square(a.goal_c.state.p_pos - a.goal_d.state.p_pos))
        rew -= np.sum(np.square(a.state.p_pos - a.goal_e.state.p_pos))
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = [world.agents[0].state.c]
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if agent.silent is False:
            return np.concatenate(
                [agent.state.p_vel]
                + [agent.goal_e.state.p_pos - agent.state.p_pos]
                + [agent.goal_b.color, agent.goal_d.color]
                + other_pos
                + comm
            )
        else:
            if agent.name == "agent 1":
                comm = [world.agents[0].state.c[: world.dim_c]]
            else:
                comm = [world.agents[0].state.c[world.dim_c :]]
            # print(comm)
            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos + comm)
