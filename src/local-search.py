##
## Parser
##

import os
import random

from math import sqrt
from sys import argv, float_info
from time import time
from pprint import pformat

class VRPInstance:
    def __init__(self, num_customers: int, num_vehicles: int,
                 vehicle_capacity: int, customer_demands: [int],
                 x_coordinates: [float], y_coordinates: [float]):
        self.num_customers    = num_customers
        self.num_vehicles     = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.customer_demands = customer_demands
        self.x_coordinates    = x_coordinates
        self.y_coordinates    = y_coordinates

        self.distance_lookup  = [[0 for _ in self.y_coordinates] for _ in self.x_coordinates]
        for a_index, ax in enumerate(self.x_coordinates):
            ay = self.y_coordinates[a_index]
            for b_index, bx in enumerate(self.y_coordinates):
                by   = self.y_coordinates[b_index]
                dist = sqrt(((ax - bx) ** 2) + ((ay - by) ** 2))
                self.distance_lookup[a_index][b_index] = dist

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def get_initial_solution(self):
        vehicle_routes = [[index] for index, _ in enumerate(self.customer_demands) if index != 0]
        return Solution(vehicle_routes)

    def get_distance_between_customers(self, a_index: int, b_index: int):
        return self.distance_lookup[a_index][b_index]

    def get_route_capacity(self, route: [int]):
        capacity = 0
        for loc in route:
            capacity += self.customer_demands[loc]
        return capacity


def parser(file_name: str):
    num_customers    = 0
    num_vehicles     = 0
    vehicle_capacity = 0
    customer_demands = []
    x_coordinates    = []
    y_coordinates    = []

    with open(file_name, "r") as file:
        for line_num, line_string in enumerate(file):
            # Parse out tokens in the line:
            tokens = line_string.strip().split(" ")
            if len(tokens) != 3:
                break

            # Parse the tokens in the line:
            if line_num == 0:
                num_customers    = int(tokens[0])
                num_vehicles     = int(tokens[1])
                vehicle_capacity = int(tokens[2])
            else:
                customer_demands.append(int(tokens[0]))
                x_coordinates.append(float(tokens[1]))
                y_coordinates.append(float(tokens[2]))

    return VRPInstance(num_customers, num_vehicles, vehicle_capacity,
                       customer_demands, x_coordinates, y_coordinates)


##
## Solver
##

class Solution:
    def __init__(self, initial_routes: [[int]]):
        self.vehicle_routes = initial_routes

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def format_routes_string(self):
        res = ""
        for route in self.vehicle_routes:
            res += " 0"
            for loc in route:
                res += " "
                res += str(loc)
            res += " 0"
        return res

def local_search(objective_function, proposal_function, initial_solution,
                 acceptance_epsilon, improvement_time, improvement_delta):
    """
    General-purposal local search minimiziation function.

    @param objective_function: The function to minimize.
    @param proposal_function: The function that gives us new solutions.
    @param initial_solution: The initial Solution to start minimizing from.
    @param acceptance_epsilon: A value for promoting exploration vs
        exploitation. New Solutions are accepted if they improve the objective
        function's value, or if they do not make it worse than the value of
        this parameter.
    @param improvement_time: The amount of time (in seconds) we're willing to
        wait before we see a "significant" improvement, where the "significance"
        of a Solution's contribution to the objective function is defined by
        improvement_delta.
    @param improvement_delta: The minimum amount of improvement that needs to
        happen in a given time period of improvement_time before we
        terminate.
    """

    initial_objective = objective_function(initial_solution)
    print(initial_objective)

    current_solution  = initial_solution
    current_objective = initial_objective

    best_solution  = initial_solution
    best_objective = initial_objective

    since_last_improvement = time()

    while ((time() < since_last_improvement + improvement_time) and
            (initial_objective - current_objective < improvement_delta)):
        proposal_solution  = proposal_function(current_solution)
        proposal_objective = objective_function(proposal_solution)

        if proposal_objective < current_objective + acceptance_epsilon:
            current_solution  = proposal_solution
            current_objective = proposal_objective

            if current_objective < best_objective: # minimize:
                best_solution  = current_solution
                best_objective = current_objective

                print(best_objective)

                if initial_objective - current_objective >= improvement_delta:
                    initial_objective      = current_objective
                    since_last_improvement = time()

    return best_solution


def objective(proposal_solution: Solution, vrp_instance: VRPInstance):
    distance = 0
    for route in proposal_solution.vehicle_routes:
        if vrp_instance.get_route_capacity(route) > vrp_instance.vehicle_capacity:
            return float_info.max

        distance += vrp_instance.get_distance_between_customers(0, route[0])
        for a, b in zip(route, route[1:]):
            distance += vrp_instance.get_distance_between_customers(a, b)
        distance += vrp_instance.get_distance_between_customers(route[-1], 0)

    return distance


##
## Heuristics
##
## Source: https://www.academia.edu/2754980/Guided_local_search_for_the_vehicle
## _routing_problem
##

def proposal_greedy_mix(x: Solution, vrp_instance: VRPInstance):
    a = proposal_two_opt_swap(Solution([a[:] for a in x.vehicle_routes]))
    b = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
    c = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
    d = proposal_cross_routes(Solution([a[:] for a in x.vehicle_routes]))

    a_obj = objective(a, vrp_instance)
    b_obj = objective(b, vrp_instance)
    c_obj = objective(c, vrp_instance)
    d_obj = objective(d, vrp_instance)

    return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]


def proposal_stochastic_greedy(x: Solution, vrp_instance: VRPInstance):
    """
    Randomly chooses a proposal heuristic and returns the result.
    """
    rand = random.randint(0, 3)
    if rand == 0:
        a = proposal_two_opt_swap(Solution([a[:] for a in x.vehicle_routes]))
        b = proposal_two_opt_swap(Solution([a[:] for a in x.vehicle_routes]))
        c = proposal_two_opt_swap(Solution([a[:] for a in x.vehicle_routes]))
        d = proposal_two_opt_swap(Solution([a[:] for a in x.vehicle_routes]))

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]
    elif rand == 1:
        a = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
        b = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
        c = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
        d = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]
    elif rand == 2:
        a = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
        b = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
        c = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
        d = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]
    elif rand == 3:
        a = proposal_cross_routes(Solution([a[:] for a in x.vehicle_routes]))
        b = proposal_cross_routes(Solution([a[:] for a in x.vehicle_routes]))
        c = proposal_cross_routes(Solution([a[:] for a in x.vehicle_routes]))
        d = proposal_cross_routes(Solution([a[:] for a in x.vehicle_routes]))

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]


def proposal_stochastic_mix(x: Solution):
    """
    Randomly chooses a proposal heuristic and returns the result.
    """
    temp = Solution([a[:] for a in x.vehicle_routes])
    rand = random.randint(0, 3)
    if rand == 0:
        return proposal_two_opt_swap(temp)
    elif rand == 1:
        return proposal_relocate_customer(temp)
    elif rand == 2:
        return proposal_exchange_customers(temp)
    elif rand == 3:
        return proposal_cross_routes(temp)

def proposal_two_opt_swap(x: Solution):
    """
    Performs a two-opt swap within a single vehicle route.
    """
    index = random.randint(0, len(x.vehicle_routes) - 1)
    route = x.vehicle_routes[index]

    if len(route) < 2:
        return x
    segment_length = random.randint(2, len(route))

    segment_start = random.randint(0, len(route) - segment_length)
    segment_end   = segment_length + segment_start

    route[segment_start:segment_end] = route[segment_start:segment_end][::-1]
    x.vehicle_routes[index] = route
    return x

def proposal_relocate_customer(x: Solution):
    """
    Relocates a customer from a route A to another route B.
    """
    route_num_a = random.randint(0, len(x.vehicle_routes) - 1)
    route_num_b = random.randint(0, len(x.vehicle_routes))

    if route_num_a == route_num_b:
        return x

    if route_num_b == len(x.vehicle_routes):
        x.vehicle_routes.append([])

    route_a    = x.vehicle_routes[route_num_a]
    route_b    = x.vehicle_routes[route_num_b]

    index_a    = random.randint(0, len(route_a) - 1)
    customer_a = route_a[index_a]
    del route_a[index_a]

    route_b.insert(random.randint(0, len(route_b)), customer_a)
    x.vehicle_routes[route_num_b] = route_b

    if len(route_a) == 0:
        del x.vehicle_routes[route_num_a]

    return x

def proposal_exchange_customers(x: Solution):
    """
    Swaps a customer A* from a route A with a customer B* from another route B.
    """
    route_num_a = random.randint(0, len(x.vehicle_routes) - 1)
    route_num_b = random.randint(0, len(x.vehicle_routes) - 1)

    if route_num_a == route_num_b:
        return x

    route_a = x.vehicle_routes[route_num_a]
    route_b = x.vehicle_routes[route_num_b]
    index_a = random.randint(0, len(route_a) - 1)
    index_b = random.randint(0, len(route_b) - 1)

    customer_a = route_a[index_a]
    customer_b = route_b[index_b]

    route_a[index_a] = customer_b
    route_b[index_b] = customer_a

    x.vehicle_routes[route_num_a] = route_a
    x.vehicle_routes[route_num_b] = route_b

    return x

def proposal_cross_routes(x: Solution):
    """
    Swaps the end portions of two vehicle routes.
    """
    route_num_a = random.randint(0, len(x.vehicle_routes) - 1)
    route_num_b = random.randint(0, len(x.vehicle_routes) - 1)

    if route_num_a == route_num_b:
        return x

    route_a = x.vehicle_routes[route_num_a]
    route_b = x.vehicle_routes[route_num_b]
    index_a = random.randint(0, len(route_a) - 1)
    index_b = random.randint(0, len(route_b) - 1)

    segment_a = route_a[index_a:len(route_a)]
    segment_b = route_b[index_b:len(route_b)]

    x.vehicle_routes[route_num_a] = route_a[:index_a]
    x.vehicle_routes[route_num_b] = route_b[:index_b]

    x.vehicle_routes[route_num_a].extend(segment_b)
    x.vehicle_routes[route_num_b].extend(segment_a)

    if len(x.vehicle_routes[route_num_a]) == 0:
        del x.vehicle_routes[route_num_a]
    if len(x.vehicle_routes[route_num_b]) == 0:
        del x.vehicle_routes[route_num_b]

    return x


##
## Main
##

def main():
    # Parse command-line arguments:
    if len(argv) != 2:
        print("Usage: " + argv[0] + "<inst-file>")
        exit(1)
    file_name = argv[1]

    # Parse instance:
    vrp_instance      = parser(file_name)
    initial_solution  = vrp_instance.get_initial_solution()
    initial_objective = objective(initial_solution, vrp_instance)

    epsilon_schedule = []
    for i in range(1, vrp_instance.num_customers):
        next_epsilon = initial_objective / (3 ** i)
        if next_epsilon < 1:
            break
        epsilon_schedule.append(next_epsilon)
    epsilon_schedule.append(1)
    epsilon_schedule.append(5)
    epsilon_schedule.append(3)
    epsilon_schedule.append(1)
    print(epsilon_schedule)


    # Solve instance:
    start_time = time()

    annealed_solution  = initial_solution
    for epsilon in epsilon_schedule:
        print("Epsilon: " + str(epsilon))
        print("Current Objective: " + str(objective(annealed_solution, vrp_instance)))
        annealed_solution = local_search(lambda x: objective(x, vrp_instance),
                                         lambda x: proposal_stochastic_greedy(x, vrp_instance),
                                         annealed_solution, epsilon, 7, 3) # float_info.max

    end_time     = time()
    elapsed_time = end_time - start_time

    # Get the optimized solution answers:
    objective_value = objective(annealed_solution, vrp_instance)
    solution_string = annealed_solution.format_routes_string()

    # Print out in the expected format:
    print("Instance: " + os.path.basename(file_name) +
          " Time: " + "{:.2f}".format(elapsed_time) +
          " Result: " + "{:.2f}".format(objective_value) + " Solution:" + solution_string)

main()
