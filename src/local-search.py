import os
import random

from math import sqrt
from sys import argv, float_info
from time import time
from pprint import pformat

##
## Parser
##

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
        for customer_a in range(0, self.num_customers):
            for customer_b in range(0, self.num_customers):
                x_sqr = (self.x_coordinates[customer_a] - self.x_coordinates[customer_b]) ** 2.0
                y_sqr = (self.y_coordinates[customer_a] - self.y_coordinates[customer_b]) ** 2.0
                self.distance_lookup[customer_a][customer_b] = sqrt(x_sqr + y_sqr)

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def get_initial_solution(self):
        """
        Returns an "initial solution" based on the contents of the VRPInstance.

        This uses a first-fit descending greedy algorithm that essentially acts
        as a "estimate" bin packing algorithm. If, in the process of this
        algorithm, we are unable to assign a given customer to any route, we
        assign it to a "ghost" vehicle, which acts as a temporary container for
        unassigned customers.

        It is up to the solver to move customers out of the "ghost" vehicle in
        order to find a feasible solution.
        """
        # Keep some variables for keeping track of routes and demands met by
        # each route:
        vehicle_routes   = [[] for _ in range(0, self.num_vehicles)]
        vehicle_demands  = [0 for _ in range(0, self.num_vehicles)]

        # Create a list of tuples where the first element is the demand of a
        # customer and the second element is the "index" (customer #) of that
        # customer, then sort them in demand-descending order:
        tuples = [(demand, index) for index, demand in enumerate(self.customer_demands)]
        tuples = sorted(tuples, key=lambda tup: tup[0])[::-1]

        # Iterate over all of the tuples in demand-descending order:
        ghost_route = []
        for (demand, customer) in tuples:
            # Skip the first "customer" (which is actually the factory:)
            if customer == 0:
                continue

            # Attempt to assign this customer into the first route that has
            # enough capacity to hold it:
            assigned = False
            for v in range(0, self.num_vehicles):
                if vehicle_demands[v] + demand <= self.vehicle_capacity:
                    vehicle_demands[v] += demand
                    vehicle_routes[v].append(customer)
                    assigned = True
                    break

            # If we weren't able to assign this customer, then let's add it to
            # the "ghost" vehicle
            if not assigned:
                print(str(customer) + " not assigned! (" + str(demand) + ")")
                ghost_route.append(customer)

        # Add the ghost vehicle to the routes:
        vehicle_routes.append(ghost_route)

        # Return a Solution instance containing the initial routes:
        return Solution(vehicle_routes)

    def get_distance_between_customers(self, a_index: int, b_index: int):
        """
        Consumes two customer numbers and returns the distance between those
        two customers.
        """
        return self.distance_lookup[a_index][b_index]

    def get_route_capacity(self, route: [int]):
        """
        Consumes a vehicle route consisting of customer numbers. Returns the
        total demand of all customers in this route.
        """
        capacity = 0
        for loc in route:
            capacity += self.customer_demands[loc]
        return capacity


def parser(file_name: str):
    """
    Consumes a file containing a .vrp instance and parses it into a VRPInstance
    for use in the local search solver.
    """
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
                # The first line contains problem metadata:
                num_customers    = int(tokens[0])
                num_vehicles     = int(tokens[1])
                vehicle_capacity = int(tokens[2])
            else:
                # All other lines correspond to demands and coordinates:
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
        """
        Prints out the vehicle routes on a single line as specified in the
        project specification for use in the results.log file.
        """
        assert(len(self.vehicle_routes[-1]) == 0)
        res = ""
        for route in self.vehicle_routes[:-1]:
            res += " 0"
            for loc in route:
                res += " "
                res += str(loc)
            res += " 0"
        return res

    def get_solution_file(self, output_file, objective_value):
        """
        Saves the vehicle routes to a file at the specified `output_file`
        filename in the format specified in the project specification for use
        with the route visualizer.
        """
        assert(len(self.vehicle_routes[-1]) == 0)
        res = "{:.2f}".format(objective_value) + " 0\n"
        for route in self.vehicle_routes[:-1]:
            res += "0"
            for loc in route:
                res += " "
                res += str(loc)
            res += " 0\n"

        with open(output_file, "w") as file:
            file.write(res)

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
    # Calculate the initial objective value:
    initial_objective = objective_function(initial_solution)

    # Keep some variables for keeping track of the current (accepted) solution:
    current_solution  = initial_solution
    current_objective = initial_objective

    # Keep some variables for keeping track of the best solution we've seen so
    # far:
    best_solution  = initial_solution
    best_objective = initial_objective

    # Iterate until we stop making enough improvements over a certain period
    # of time (based on the improvement_time and improvement_delta arguments):
    since_last_improvement = time()
    while ((time() < since_last_improvement + improvement_time) and
            (initial_objective - current_objective < improvement_delta)):
        # Propose a new solution and get its objective value:
        proposal_solution  = proposal_function(current_solution)
        proposal_objective = objective_function(proposal_solution)

        # Check if this meets our acceptance criterion:
        if proposal_objective - current_objective < acceptance_epsilon:
            # If so, update the current (accepted) solution variables:
            current_solution  = proposal_solution
            current_objective = proposal_objective

            # Check if it's a new best solution:
            if current_objective < best_objective:
                # If so, update the best solution variables:
                best_solution  = current_solution
                best_objective = current_objective
                print(best_objective)

                # If we've improved enough (based on the improvement_delta
                # argument), restart our expiration timer on this local search
                # run:
                if initial_objective - current_objective >= improvement_delta:
                    initial_objective      = current_objective
                    since_last_improvement = time()

    # Return the best solution we've seen so far:
    return best_solution


def objective(proposal_solution: Solution, vrp_instance: VRPInstance):
    """
    The objective function for the vehicle routing problem. In "normal" cases,
    returns a value equal to the total distance travelled by all routes in the
    input Solution.

    If any of a given Solution's routes do not satisfy the `vehicle_capacity`
    constraint, returns float_info.max as a penalty value.

    If any customers are currently being serviced by the "ghost" vehicle, the
    distance traversed by the "ghost" vehicle is multiplied by a large constant
    as a penalty. This incentivizes the local search algorithm to move as many
    things out of the "ghost" vehicle as possible as to lower the objective
    function's value.
    """
    # Loop over every single route and determine how much distance it travels:
    distances = []
    for index, route in enumerate(proposal_solution.vehicle_routes):
        # Check if this is a valid vehicle (that is, not the "ghost" vehicle:)
        if index < vrp_instance.num_vehicles:
            # Check if the solution satisfies the capacity constraint. If it
            # doesn't, return a maximum float penalty value:
            if vrp_instance.get_route_capacity(route) > vrp_instance.vehicle_capacity:
                return float_info.max

        # Otherwise, compute the total distance of this route and add it to our
        # tracking array:
        route_distance = 0
        if len(route) > 0:
            route_distance += vrp_instance.get_distance_between_customers(0, route[0])
            for a, b in zip(route, route[1:]):
                route_distance += vrp_instance.get_distance_between_customers(a, b)
            route_distance += vrp_instance.get_distance_between_customers(route[-1], 0)
        distances.append(route_distance)

    # Assign a penalty to customer locations that are located within the
    # "ghost" vehicle (this incentivizes the local search algorithm to move
    # as many things out of the "ghost" vehicle as possible):
    distances[-1] *= 100000

    # Return the sum of all of the route distances:
    return sum(distances)


##
## Heuristics
##
## Source: https://www.academia.edu/2754980/Guided_local_search_for_the_vehicle
## _routing_problem
##

def proposal_greedy_mix(x: Solution, vrp_instance: VRPInstance):
    """
    Tries all four proposal functions, then returns the best result.
    """
    a = proposal_two_opt_swap(Solution([a[:] for a in x.vehicle_routes]))
    b = proposal_three_opt_swap(Solution([a[:] for a in x.vehicle_routes]), vrp_instance)
    c = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
    d = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
    e = proposal_cross_routes(Solution([a[:] for a in x.vehicle_routes]))

    a_obj = objective(a, vrp_instance)
    b_obj = objective(b, vrp_instance)
    c_obj = objective(c, vrp_instance)
    d_obj = objective(d, vrp_instance)
    e_obj = objective(e, vrp_instance)

    return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d), (e_obj, e)], key=lambda tup: tup[0])[0][1]

def proposal_stochastic_greedy(x: Solution, vrp_instance: VRPInstance):
    """
    Tries all four proposal functions several times, then returns the best
    result.
    """
    rand = random.randint(0, 4)
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
    if rand == 1:
        a = proposal_three_opt_swap(Solution([a[:] for a in x.vehicle_routes]), vrp_instance)
        b = proposal_three_opt_swap(Solution([a[:] for a in x.vehicle_routes]), vrp_instance)
        c = proposal_three_opt_swap(Solution([a[:] for a in x.vehicle_routes]), vrp_instance)
        d = proposal_three_opt_swap(Solution([a[:] for a in x.vehicle_routes]), vrp_instance)

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]
    elif rand == 2:
        a = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
        b = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
        c = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))
        d = proposal_relocate_customer(Solution([a[:] for a in x.vehicle_routes]))

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]
    elif rand == 3:
        a = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
        b = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
        c = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))
        d = proposal_exchange_customers(Solution([a[:] for a in x.vehicle_routes]))

        a_obj = objective(a, vrp_instance)
        b_obj = objective(b, vrp_instance)
        c_obj = objective(c, vrp_instance)
        d_obj = objective(d, vrp_instance)

        return sorted([(a_obj, a), (b_obj, b), (c_obj, c), (d_obj, d)], key=lambda tup: tup[0])[0][1]
    elif rand == 4:
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

def proposal_three_opt_swap(x: Solution, vrp_instance: VRPInstance):
    """
    Performs a three-opt swap within a single vehicle route.
    """
    index = random.randint(0, len(x.vehicle_routes) - 1)
    route = x.vehicle_routes[index]

    if len(route) < 6:
        return x

    a = random.randint(0, len(route) - 6)
    b = random.randint(a, len(route) - 5)

    c = random.randint(b, len(route) - 4)
    d = random.randint(c, len(route) - 3)

    e = random.randint(d, len(route) - 2)
    f = random.randint(e, len(route) - 1)

    d0 = vrp_instance.get_distance_between_customers(a, b) + \
         vrp_instance.get_distance_between_customers(c, d) + \
         vrp_instance.get_distance_between_customers(e, f)
    d1 = vrp_instance.get_distance_between_customers(a, c) + \
         vrp_instance.get_distance_between_customers(b, d) + \
         vrp_instance.get_distance_between_customers(e, f)
    d2 = vrp_instance.get_distance_between_customers(a, b) + \
         vrp_instance.get_distance_between_customers(c, e) + \
         vrp_instance.get_distance_between_customers(d, f)
    d3 = vrp_instance.get_distance_between_customers(a, d) + \
         vrp_instance.get_distance_between_customers(e, b) + \
         vrp_instance.get_distance_between_customers(c, f)
    d4 = vrp_instance.get_distance_between_customers(f, b) + \
         vrp_instance.get_distance_between_customers(c, d) + \
         vrp_instance.get_distance_between_customers(e, a)

    if d0 > d1:
        route[b:d] = route[b:d][::-1]
    elif d0 > d2:
        route[d:f] = route[d:f][::-1]
    elif d0 > d4:
        route[b:f] = route[b:f][::-1]
    elif d0 > d3:
        route[b:f] = route[d:f] + route[b:d]

    x.vehicle_routes[index] = route

    return x


def proposal_relocate_customer(x: Solution):
    """
    Relocates a customer from a route A to another route B.
    """
    route_num_a = random.randint(0, len(x.vehicle_routes) - 1)
    route_num_b = random.randint(0, len(x.vehicle_routes) - 1)

    if route_num_a == route_num_b:
        return x

    route_a    = x.vehicle_routes[route_num_a]
    route_b    = x.vehicle_routes[route_num_b]

    if len(route_a) > 0:
        index_a    = random.randint(0, len(route_a) - 1)
        customer_a = route_a[index_a]
        del route_a[index_a]

        route_b.insert(random.randint(0, len(route_b)), customer_a)
        x.vehicle_routes[route_num_b] = route_b

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

    if len(route_a) > 0 and len(route_b) > 0:
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

    if len(route_a) > 0 and len(route_b) > 0:
        index_a = random.randint(0, len(route_a) - 1)
        index_b = random.randint(0, len(route_b) - 1)

        segment_a = route_a[index_a:len(route_a)]
        segment_b = route_b[index_b:len(route_b)]

        x.vehicle_routes[route_num_a] = route_a[:index_a]
        x.vehicle_routes[route_num_b] = route_b[:index_b]

        x.vehicle_routes[route_num_a].extend(segment_b)
        x.vehicle_routes[route_num_b].extend(segment_a)

    return x

def proposal_stochastic_route_swapping(x: Solution, vrp_instance: VRPInstance):
    """
    Randomly chooses a proposal heuristic related to improving single vehicle
    routes and returns the result.
    """
    temp = Solution([a[:] for a in x.vehicle_routes])
    rand = random.randint(0, 1)
    if rand == 0:
        return proposal_three_opt_swap(temp, vrp_instance)
    elif rand == 1:
        return proposal_two_opt_swap(temp)

def proposal_reorder_customer_in_route(x: Solution):
    """
    Moves a customer within a route to another location in the route.
    """
    route_num = random.randint(0, len(x.vehicle_routes) - 1)
    route     = x.vehicle_routes[route_num]

    if len(route) > 0:
        loc_index  = random.randint(0, len(route) - 1)
        customer = route[loc_index]
        del route[loc_index]

        route.insert(random.randint(0, len(route)), customer)
        x.vehicle_routes[route_num] = route

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

    # Set up some annealing schedules to use during our local search iterations:
    epsilon_schedule     = [1000, 100, 50, 10, 5,  2,  1.5, 1,  0.5, 0.1]
    improvement_schedule = [100,  10,  8,  8,  7,  6,  3,   3,  1,   0.5]
    timeout_schedule     = [2,    4,   6,  12, 8,  6,  5,   4,  4,   4]

    # Keep some variables for tracking state over iterations:
    iters_since_change = 0
    annealed_solution  = initial_solution

    # Start solving the instance using local search:
    start_time = time()

    # Loop over every scheduling tuple:
    for iter_num in range(0, len(epsilon_schedule)):
        # Get the hyperparameter for this scheduling instance:
        epsilon           = epsilon_schedule[iter_num]
        improvement_delta = improvement_schedule[iter_num]
        timeout           = timeout_schedule[iter_num]

        # Print out some useful information regarding the current scheduling
        # state:
        current_objective = objective(annealed_solution, vrp_instance)
        print("Epsilon: " + str(epsilon))
        print("Current Objective: " + str(current_objective))

        # Some of the "big" epsilons only are applicable to large problems, so
        # this skips big epsilons when they're probably not going to help much:
        if (initial_objective / 3) < epsilon:
            continue

        # Apply local search to the current solution:
        next_annealed = local_search(lambda x: objective(x, vrp_instance),
                                     lambda x: proposal_stochastic_greedy(x, vrp_instance),
                                     annealed_solution, epsilon, timeout, improvement_delta)

        prev_objective = objective(annealed_solution, vrp_instance)
        next_objective = objective(next_annealed, vrp_instance)

        annealed_solution = next_annealed

        # If we haven't changed our objective value in the last three
        # iterations, terminate the local search altogether:
        if prev_objective - next_objective < 0.00001:
            iters_since_change += 1
            if iters_since_change > 2:
                break
        else:
            iters_since_change = 0

    # print("Done with annealing. Going to two-opt swap!")

    # epsilon_schedule = [5, 3, 1]
    # for epsilon in epsilon_schedule:
    #     current_objective = objective(annealed_solution, vrp_instance)
    #     print("Epsilon: " + str(epsilon))
    #     print("Current Objective: " + str(current_objective))

    #     annealed_solution = local_search(lambda x: objective(x, vrp_instance),
    #                                  lambda x: proposal_stochastic_route_swapping(x, vrp_instance),
    #                                  annealed_solution, epsilon, 5, 1)


    end_time     = time()
    elapsed_time = end_time - start_time

    # Get the optimized solution answers:
    objective_value = objective(annealed_solution, vrp_instance)
    solution_string = annealed_solution.format_routes_string()

    # Print out in the expected format:
    print("Instance: " + os.path.basename(file_name) +
          " Time: " + "{:.2f}".format(elapsed_time) +
          " Result: " + "{:.2f}".format(objective_value) + " Solution:" + solution_string)

    # Save the solution to a solution file:
    solution_filename = "./sol/" + os.path.basename(file_name) + ".sol"
    annealed_solution.get_solution_file(solution_filename, objective_value)

main()
