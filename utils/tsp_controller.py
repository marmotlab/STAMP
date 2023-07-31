import math
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class TSPSolver:
    def __init__(self):
        self.magnify = 100000  # ease numerical calculation
        self.coords = None

    def create_data_model(self, coords):
        """Stores the data for the problem."""
        data = dict()
        # Locations in block units
        data['locations'] = coords * self.magnify  # yapf: disable
        data['num_vehicles'] = 1
        data['starts'] = [1]
        data['ends'] = [0]
        return data

    def compute_euclidean_distance_matrix(self, locations):
        """Creates callback to return distance between points."""
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    # Euclidean distance
                    distances[from_counter][to_counter] = (int(math.hypot((from_node[0] - to_node[0]),(from_node[1] - to_node[1]))))
        return distances

    @staticmethod
    def print_solution(manager, routing, solution):
        """Prints solution on console."""
        route = [1]
        # print('Objective: {}'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        # plan_output = 'Route: '
        route_distance = 0
        while not routing.IsEnd(index):
            # plan_output += ' {} -'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            route.append(manager.IndexToNode(index))
        # plan_output += ' {}\n'.format(manager.IndexToNode(index))
        # print(plan_output)
        # plan_output += 'Objective: {}m\n'.format(route_distance)
        return route

    def run_solver(self, coords):
        """Entry point of the program."""
        # Instantiate the data problem.
        self.coords = coords * self.magnify
        data = self.create_data_model(self.coords)
        distance_matrix = self.compute_euclidean_distance_matrix(data['locations'])
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                               data['num_vehicles'], data['starts'], data['ends'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            route = self.print_solution(manager, routing, solution)
        return route


if __name__ == '__main__':
    pts = np.random.rand(100, 2)
    tsp = TSPSolver()
    tsp.run_solver(pts)