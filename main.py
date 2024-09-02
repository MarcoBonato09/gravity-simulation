# This allows us to type hint a class inside its own definition
from __future__ import annotations

# Literal[a, b, c, ...] is a type hint meaning either a or b or c and so on
from typing import Literal

import math # Used for its sin, cos, atan2, sqrt, ceil, and degrees functions
import os # Used to forcefully exit the program
import threading # Used to open the back-end main loop in a separate thread
import pygame # Used to draw the grid, bodies, orbit preview, etc.
import tkinter # Used to create the user interface window
from tkinter.ttk import Style # Used to change the appearance of widgets
from tkinter import StringVar # Used to get and set values shown in widgets

# This block imports all the tkinter widgets used in the user interface window
from tkinter import Canvas, Frame, Button, Label, Entry, Scrollbar
from tkinter.ttk import Combobox, Notebook, Treeview


class Vector:
    """This class represents a 2d vector and is used throughout this program.
    The components, magnitude and direction of the vector are stored."""
        
    def __init__(self,
                 x_component: int|float|None = None,
                 y_component: int|float|None = None,
                 magnitude: int|float|None = None,
                 direction: int|float|None = None) -> None:
        """This function initializes this vector object, given 
        its components or its magnitude and direction."""
        
        # The __ before attribute names means that they are private
        self.__x_component = x_component
        self.__y_component = y_component
        self.__magnitude = magnitude
        self.__direction = direction
        
        # We need either both components or a magnitude and direction
        # Once we have either we calculate the rest of the attributes
        if x_component is not None and y_component is not None:
            self.recalculate_polar_form()
        elif magnitude is not None and direction is not None:
            self.recalculate_components()
                    
                        
    def recalculate_components(self) -> None:
        """This function recalculates the components of 
        this vector using its magnitude and direction."""
        
        self.__x_component = self.__magnitude*math.cos(self.__direction)
        self.__y_component = self.__magnitude*math.sin(self.__direction)
             
             
    def recalculate_polar_form(self) -> None:
        """This function recalculates the magnitude and 
        direction of this vector using its components."""
        
        # Here we use the Pythagoras theorem to find this vector's magnitude
        self.__magnitude = math.sqrt(self.__x_component**2 + self.__y_component**2)
        
        # The atan2 function returns a vector's direction given its components
        self.__direction = math.atan2(self.__y_component, self.__x_component)


    # We use @property to declare dynamically calculated attributes
    # These are not stored but calculated from other attributes when retrieved
    # They are called automatically when <class>.dynamic_attribute is called
    # For example, <vector>.components calls the components function below
    
    @property
    def components(self) -> tuple[int|float, int|float]:
        """This function returns the components of this vector , 
        vector as a tuple, starting with the x component."""
        
        return (self.x_component, self.y_component)
    
    
    # These next @property functions are used as getters for our private attributes
    
    @property
    def x_component(self) -> int|float:
        """This function returns the x component of this vector."""
        
        return self.__x_component
    

    @property
    def y_component(self) -> int|float:
        """This function returns the y component of this vector."""
        
        return self.__y_component
    

    @property
    def magnitude(self) -> int|float:
        """This function returns the magnitude of this vector."""
        
        return self.__magnitude
    
    
    @property
    def direction(self) -> int|float:
        """This function returns the direction of this vector in radians.
        Remember that this is calculated anticlockwise starting East."""
        
        return self.__direction
    
    
    # We use @attribute.setter to give dynamically calculated attributes setters
    # We do this by updating the attributes used to calculate the dynamic attributes
    # They are called automatically when calling <class>.dynamic_attribute = ...
    # For example, <vector>.x_component = ... calls the x_component function below
    # We use this to give our private attributes setters, so that we can call one of 
    # the recalculate_ functions automatically when a private attribute is set
    
    @x_component.setter
    def x_component(self, new_x_component: int|float) -> None:
        """This function sets the x component of this vector 
        to the given value and recalculates its polar form."""
        
        self.__x_component = new_x_component
        self.recalculate_polar_form()  
        
          
    @y_component.setter
    def y_component(self, new_y_component: int|float) -> None:
        """This function sets the y component of this vector to the 
        given value and recalculates its magnitude and direction."""
        
        self.__y_component = new_y_component
        self.recalculate_polar_form()    
        
        
    @magnitude.setter
    def magnitude(self, new_magnitude: int|float) -> None:
        """This function sets the magnitude of this vector to
        the given value and recalculates its components."""
        
        # As a note, the user CAN input a negative magnitude
        # This will update this vector's magnitude AND reverse its direction
        
        self.__magnitude = new_magnitude
        self.recalculate_components()    
        
        
    @direction.setter
    def direction(self, new_direction: int|float) -> None:
        """This function sets the direction of this vector to the 
        given value (in radians) and recalculates its components."""
        
        self.__direction = new_direction
        self.recalculate_components()
            
            
    # These next methods are additional helper functions
              
    def soften(self, softening_factor: int|float) -> None:
        """This function softens this vector using a given softening factor.
        By definition, this means settings its magnitude to 
        √(old_magnitude² + softening_factor²)."""
        
        self.magnitude = math.sqrt(self.magnitude**2 + softening_factor**2)
    
    
    def copy(self) -> Vector:
        """This function returns a copy of this vector."""
        
        return Vector(self.x_component, self.y_component)
    
    
    # The following methods with __ around their name are called magic methods
    # Here they are used to define mathematical operations on vectors
    # They are called automatically when an operation is performed on a vector
    # For example, <vector_1> + <vector_2> calls the __add__ method
        
    def __add__(self, other_vector: Vector) -> Vector:
        """This function defines the addition operation for a vector.
        When a vector is added to another their components are summed."""
        
        return Vector(self.x_component + other_vector.x_component, 
                      self.y_component + other_vector.y_component)
    
    
    def __sub__(self, other_vector: Vector) -> Vector:
        """This function defines the subtraction operation for a vector. When a vector
        is subtracted from another their components are subtracted from each other."""
                
        return Vector(self.x_component - other_vector.x_component, 
                      self.y_component - other_vector.y_component)
    
    
    def __mul__(self, multiplication_value: int|float|Vector) -> Vector:
        """This function defines the multiplication operation for a vector.
        When a vector is multiplied by a numerical value its magnitude is 
        multiplied by this value. Otherwise, if it is multiplied by a vector 
        then the dot product of the two vectors is returned."""
        
        # isinstance(x, a|b) checks if x is of type a or b
        if isinstance(multiplication_value, int|float):
            return Vector(magnitude = self.magnitude*multiplication_value, 
                          direction = self.direction)
        else:
            other_vector = multiplication_value
            
            # This formula below is the definition of the dot product
            # We sum the multiples of the vectors' respective components
            dot_product = (self.x_component*other_vector.x_component 
                           + self.y_component*other_vector.y_component)
            return dot_product
        
        
    def __rmul__(self, multiplication_value: int|float|Vector) -> Vector:
        """This is the same as __mul__ but for when the multiplication value 
        is to the left of the vector. For example, 5*<vector> calls __rmul__ 
        instead of __mul__. For our class, order of multiplication does not 
        matter, so rmul is equivalent to mul."""
        
        return self.__mul__(multiplication_value)
        
    
    def __truediv__(self, division_factor: int|float) -> Vector:
        """This function defines the division operation for a vector.
        When a vector is divided by a numerical value its magnitude 
        is divided by this value."""
                
        return Vector(magnitude = self.magnitude/division_factor, 
                      direction = self.direction)
    
    
    def __floordiv__(self, division_factor: int|float) -> Vector:
        """This function defines the floor division operation for a vector.
        When a vector is floor divided by a numerical value its components are 
        floor divided by this value."""
        
        return Vector(self.x_component//division_factor, 
                      self.y_component//division_factor)
        
    
    def __mod__(self, modulo_factor: int|float) -> Vector:
        """This function defines the modulo operation for a vector. A vector modulo
        some number sets the components of that vector modulo that number."""
        
        return Vector(self.x_component%modulo_factor, 
                      self.y_component%modulo_factor)    
    
    
    def __neg__(self) -> Vector:
        """This function defines the - symbol in front of a vector.
        Specifically, -<vector> calls this function. 
        This returns the vector with its components inverted."""
        
        return Vector(-self.x_component, -self.y_component)
        

class Body:
    """Bodies are the basic items that move in the simulation, representing 
    objects such as planets or stars. They are circular with uniformly 
    distributed masses and have numerous attributes introduced below."""
    
    def __init__(self, 
                 id: str, 
                 mass: int|float, 
                 radius: int|float, 
                 position: Vector) -> None:
        """This function initializes this body's attributes to the given values.
        Other attributes such as velocity or acceleration are set to a default
        value and can then be changed by the user. All attributes use standard 
        Physics units (e.g. mass in kilograms, lengths in meters, etc.)."""
                
        self.id = id
        self.mass = mass
        self.radius = radius
        self.previous_position = position
        self.position = position
        self.is_immobile = False
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.engine_acceleration = Vector(0, 0)
        
        
    def step_euler(self, time_step: int|float) -> None:
        """This function uses the Euler method to update the velocity and 
        position of this body given the time step parameter (in seconds). The 
        higher the time step parameter, the further the body will move, but 
        the less accurate its movement. The opposite is also true. 
        If the body is immobile its velocity and acceleration are set to 0."""
        
        if not self.is_immobile:
            self.previous_position = self.position.copy()
            total_acceleration = self.acceleration + self.engine_acceleration
            self.velocity += total_acceleration*time_step
            self.position += self.velocity*time_step
        else:
            self.velocity = Vector(0, 0)
            self.acceleration = Vector(0, 0)
    
            
    @property
    def kinetic_energy(self) -> float:
        """This function returns this body's kinetic energy in Joules"""
        
        return 1/2*self.mass*self.velocity.magnitude**2
    
    
    @kinetic_energy.setter
    def kinetic_energy(self, new_kinetic_energy: int|float) -> None:
        """This function sets the kinetic energy 
        of this body to the inputted value."""
        
        self.velocity.magnitude = math.sqrt(2*new_kinetic_energy/self.mass)    
    
    
    @property
    def momentum(self) -> Vector:
        """This function returns the momentum vector of this body"""
        
        return self.velocity*self.mass
    
    
    # Functions are used for angular attributes since they require parameters
    
    def angular_velocity(self, centered_body: Body, time_step: int|float) -> float:
        """This function returns the angular velocity of this body. By convention,
        it is positive if the body is rotating anticlockwise around the centered 
        body and around negative if rotating clockwise."""
        
        # Angular velocity is found by finding the change in direction of a vector
        # pointing from this body to the centered body, divided by the time taken
        
        # <position_vector_2> - <position_vector_1> gives the displacement vector 
        # from position 1 to position 2. This is used throughout this program.
        previous_angle = (centered_body.position-self.previous_position).direction
        current_angle = (centered_body.position-self.position).direction
        angular_velocity = (current_angle-previous_angle)/time_step
                
        return angular_velocity
    
    
    def angular_momentum(self, centered_body: Body) -> int|float:
        """This function returns the angular momentum of this body around the 
        centered body. This is calculated using the cross product of the body's
        position vector (measured from the centered body) and momentum vector."""
        
        position_vector = self.position - centered_body.position
        
        # This is the the body's momentum vector rotated 90 degrees
        # Vectors that are 90 degrees to other vectors are called normal vectors
        momentum_normal = Vector(self.momentum.y_component, -self.momentum.x_component)
        
        cross_product = position_vector*momentum_normal
        return cross_product
    

class BarnesHutNode:
    """This class represents one of the nodes in a Barnes-Hut quadtree. 
    It also stores the attributes of the square in space the node represents.
    This is called the square of a node."""
        
    def __init__(self,
                 contained_bodies: list[Body],
                 center: Vector,
                 width: int|float) -> None:
        """This function initializes this node's attribute variables. 
        It does not find properties such as its total mass or children. 
        That is handled by the BarnesHutTree class."""
        
        # These attributes are properties of this node's square
        self.contained_bodies = contained_bodies
        self.num_contained_bodies = len(self.contained_bodies)
        self.center = center
        self.width = width
        
        # These attributes reference the bodies contained in the square
        self.total_mass = 0
        self.max_radius = 0
        self.center_of_mass = Vector(0, 0)
        
        # This list stores this node's children (other BarnesHutNode objects)
        # The values, in order, at each position, are the top left, top right, 
        # bottom left and bottom right children respectively.
        self.children: list[BarnesHutNode|None] = [None, None, None, None]
         
        
class BarnesHutTree:
    """This class is responsible for initializing and storing the root node 
    of a Barnes-Hut quadtree. Operations such as traversal for finding 
    forces or collision detection is handled by the Simulation class."""
    
    def __init__(self, contained_bodies: list[Body]) -> None:
        """This function initializes the root node of the quadtree given a 
        list of the bodies in the simulation."""
        
        root_center, root_width = self.find_bounding_box(contained_bodies)
        self.root_node = self.build_node(contained_bodies, root_center, root_width)
        
        
    def find_bounding_box(self, contained_bodies: list[Body]) -> tuple[Vector, int]:
        """This function returns the center coordinates and width of the 
        smallest square that contains all the bodies in contained_bodies."""
        
        # First we find the minimum and maximum x and y coordinates
        # default=0 means that 0 is returned if the argument to min() is empty
        x_coordinates = [body.position.x_component for body in contained_bodies]
        y_coordinates = [body.position.y_component for body in contained_bodies]
        minimum_coordinates = Vector(min(x_coordinates, default=0), 
                                     min(y_coordinates, default=0))
        maximum_coordinates = Vector(max(x_coordinates, default=0), 
                                     max(y_coordinates, default=0))
            
        # Next we calculate the center and width of the square
        
        # The width of the square is the maximum difference between the
        # maximum and minimum coordinates in each direction. A little extra 
        # width is added (+2), since otherwise a body right on the edge of the 
        # square boundary might not be detected due to a floating point error.
        difference = maximum_coordinates - minimum_coordinates
        max_difference = max(difference.components)
        width = math.ceil(max_difference) + 2
        
        center = (minimum_coordinates+maximum_coordinates)/2
        
        return center, width
    
    
    def build_node(self, 
                   contained_bodies: list[Body], 
                   center: Vector, 
                   width: int) -> BarnesHutNode:
        """This function returns a BarnesHutNode with its attributes 
        fully initialized, given information about its square."""
                
        node = BarnesHutNode(contained_bodies, center, width)
                        
        # This list stores the bodies in each quadrant of this node's square 
        # In order, these lists store the bodies for the top left, top right, 
        # bottom left and bottom right quadrants respectively
        quadrant_bodies: list[list[Body]] = [[], [], [], []]
        
        # This loop sorts each body in contained_bodies into one of the
        # quadrants and calculates the node's attributes (except its children)
        for body in contained_bodies:
            node.max_radius = max(node.max_radius, body.radius)
            node.total_mass += body.mass
            node.center_of_mass += body.mass*body.position
            
            # This next block finds the quadrant this body is in and
            # assigns it to the respective list in quadrant_bodies
                    
            if body.position.y_component >= node.center.y_component:
                # This body is in the top part of the square
                # Now we compare x coordinates to see if it's i
                # in the left or right part of the top part
                if body.position.x_component <= node.center.x_component:
                    quadrant_bodies[0].append(body)
                else:
                    quadrant_bodies[1].append(body)
            # Otherwise it's in the bottom part of the square
            # Again we compare x coordinates to check for left or right
            elif body.position.x_component <= node.center.x_component:
                quadrant_bodies[2].append(body)
            else:
                quadrant_bodies[3].append(body)
                    
        if node.num_contained_bodies != 0:
            node.center_of_mass /= node.total_mass
        
        # Next, this block construct this node's children recursively. 
        # A node only has children if it contains more than 1 body.
        if node.num_contained_bodies > 1:
            # In order, this list stores the centers for the top left, top 
            # right, bottom left and bottom right quadrants respectively
            quadrant_centers = [
                node.center + Vector(-node.width/4, node.width/4), 
                node.center + Vector(node.width/4, node.width/4), 
                node.center + Vector(-node.width/4, -node.width/4), 
                node.center + Vector(node.width/4, -node.width/4)]
                        
            # This loop recursively calls build_node to create this node's children
            for i in range(4):
                node.children[i] = self.build_node(quadrant_bodies[i],  
                                                   quadrant_centers[i], 
                                                   node.width/2)

        return node
    

class Simulation:
    """This class is responsible for acceleration calculation, moving each 
    body and handling collisions. If there is a collision between bodies they 
    are merged. The user may choose between the Direct Sum or Barnes-Hut 
    algorithms for acceleration calculation and collision handling."""
    
    G = 6.6743e-11 # The constant used in Newton's law of gravitation
        
    def __init__(self) -> None:
        """This functions initializes the attributes of the simulation."""
        
        # This dictionary maps the id of each body to its Body object
        # This allows for quickly checking if a body exists and quickly
        # adding or removing bodies from the simulation
        self.contained_bodies: dict[str, Body] = {}

        self.time_step = 1 # The parameter used for the Euler method
        self.time_elapsed = 0
        
        # The parameter for the Barnes-Hut algorithm. Higher values mean
        # a less accuracate but faster simulation. The opposite is also true.
        self.theta = 0.5
        
        # The parameter for the smoothening technique. Higher values mean a 
        # larger tolerance to acceleration values spiking but less accuracy.
        self.epsilon = 0
        
        self.barnes_hut_tree = BarnesHutTree([])

        # These next variables store the currently used algorithms for
        # collision detection and acceleration calculation in the simulation
        self.acceleration_calculation: Literal["Direct Sum", "Barnes-Hut"] = "Direct Sum"
        self.collision_detection: Literal["Direct Sum", "Direct Sum (Optimized)",
                                          "Barnes-Hut", "None"] = "Direct Sum"
        
    
    def total_of_property(self, 
                          attribute_name: str,
                          centered_body: Body|None = None) -> int|float:
        """This function takes an attribute name and sums that attribute over 
        all bodies. For example, if attribute_name is "velocity.x_component", 
        then the x component of the total velocity of all bodies is returned."""
                
        total = 0
        if len(attribute_name.split(".")) > 1: # This means we are summing a vector
            total = Vector(0, 0) # Therefore our total needs to be a Vector object
        
        for id, body in self.contained_bodies.items():
            if attribute_name == "angular_momentum":
                total += body.angular_momentum(centered_body)
            elif attribute_name == "angular_velocity":
                total += body.angular_velocity(centered_body, self.time_step)
            elif attribute_name == "kinetic_energy":
                total += body.kinetic_energy
            else: # Otherwise, we are summing a vector property
                # In this case, we first sum the vector and then get
                # its desired attribute at the end (see line 544)
                vector_name, vector_attribute = attribute_name.split(".")
                total += getattr(body, vector_name)
                
        if isinstance(total, Vector):
            return getattr(total, vector_attribute)
        else:
            return total
                
        
    def find_unique_id(self) -> str:
        """This function returns a unique id not already in use. It returns a 
        string "n", where n is the smallest integer >= 0 such that the id "n" 
        does not already exist."""
        
        id_number = 0 # Represents the letter n explained in the docstring
        
        # This loop keeps incrementing id number until the id "<id_number>" 
        # does not already exist in the simulation
        while str(id_number) in self.contained_bodies:
            id_number += 1
            
        return str(id_number)
        
        
    def have_bodies_collided(self, body_1: Body, body_2: Body) -> bool:
        """This function returns True or False based on if two bodies have 
        collided. This also checks if they have collided inter-frame."""
                
        # a, b and c are the coefficients of the quadratic equation used to 
        # check for inter-frame collisions, shown in section 1.3.6.2
        collided_inter_frame = False
        a = (body_1.velocity - body_2.velocity).magnitude**2
        b = 2*((body_1.position-body_2.position)*(body_1.velocity-body_2.velocity))
        c = ((body_1.position-body_2.position).magnitude**2
             - (body_1.radius+body_2.radius)**2)
        determinant = b**2 - 4*a*c
        
        # This condition must be met for the quadratic equation to have roots
        if determinant >= 0 and a != 0:
            # We now find the roots of the quadratic equation
            # This is done using the quadratic formula as shown below
            root_1 = (-b+math.sqrt(determinant))/(2*a)
            root_2 = (-b-math.sqrt(determinant))/(2*a)
            lesser_root = min(root_1, root_2)
            larger_root = max(root_1, root_2)
            
            # We now check if the time step parameter is between lesser_root 
            # and larger_root. If so, the bodies have collided inter-frame.
            if lesser_root <= self.time_step <= larger_root:
                collided_inter_frame = True
        
        distance_between_bodies = (body_2.position - body_1.position).magnitude
        if distance_between_bodies < body_1.radius+body_2.radius or collided_inter_frame:
            return True
        else:
            return False
        
        
    def merge_bodies(self, body_1: Body, body_2: Body) -> Body:
        """This function takes two bodies, removes them and merges them 
        while conserving various properties. Then it adds the merged body."""
        
        merged_body_id = self.find_unique_id()
        merged_body_mass = body_1.mass+body_2.mass # This conserves mass
        
        # This new radius conserves area
        merged_body_radius = math.sqrt(body_1.radius**2 + body_2.radius**2)
                
        # This velocity of the merged body conserves momentum in the simulation
        merged_body_velocity = (body_1.velocity*body_1.mass 
                                + body_2.velocity*body_2.mass)/merged_body_mass
        
        # The merged body's position is the center of mass of the two bodies
        merged_body_position = (body_1.position*body_1.mass 
                                + body_2.position*body_2.mass)/merged_body_mass
            
        # This block creates the new body and sets its velocity
        merged_body = Body(merged_body_id, merged_body_mass, 
                           merged_body_radius, merged_body_position)
        merged_body.velocity = merged_body_velocity

        # This block removes the bodies that merged and adds the new one
        self.contained_bodies.pop(body_1.id)
        self.contained_bodies.pop(body_2.id)
        self.contained_bodies[merged_body.id] = merged_body
         
        return merged_body
                
                
    def merge_bodies_in_same_location(self) -> None:
        """This function merges bodies with the same position, which is needed
        as building the Barnes-Hut quadtree will recurse infinitely if two 
        bodies are in the same location."""
        
        # This dictionary maps coordinates to a body
        # It is used to quickly check if two bodies are in the same location
        location_dictionary: dict[tuple[int|float, int|float], Body] = {}
        
        contained_bodies_list = list(self.contained_bodies.values())
        for body in contained_bodies_list:
            if body.position.components in location_dictionary:
                # There are two bodies with the same location so we merge them
                other_body = location_dictionary[body.position.components]
                merged_body = self.merge_bodies(body, other_body)
                # We now replace the body stored at the coordinates
                # in the dictionary with the merged body
                location_dictionary[body.position.components] = merged_body
            else:
                location_dictionary[body.position.components] = body
                
          
    def direct_sum_collision_handling(self) -> None:
        """This function merges bodies that have collided using the Direct Sum method.
        This means repeatedly checking for collisions between every pair of bodies."""
                    
        merge_occured = True
        while merge_occured: # The process is repeated until no merges happen
            contained_bodies_list = list(self.contained_bodies.values())
            removed_bodies: set[Body] = set()
            merge_occured = False
            
            # Iterate over every pair of bodies and check for a collision
            for i in range(len(contained_bodies_list)):
                body_1 = contained_bodies_list[i]
                if body_1 in removed_bodies: continue # We skip over removed bodies
                
                for j in range(i+1, len(contained_bodies_list)):
                    body_2 = contained_bodies_list[j]
                    if body_2 in removed_bodies: continue
                    elif self.have_bodies_collided(body_1, body_2):
                        merge_occured = True
                        removed_bodies.add(body_1)
                        removed_bodies.add(body_2)  
                        body_1 = self.merge_bodies(body_1, body_2)


    def optimized_direct_sum_collision_handling(self) -> None:
        """This function is an optimized version of direct_sum_collision_handling,
        but with the same O(n²) time complexity. Read section 1.3.7.2 for an 
        explanation of why this version is optimized."""
        
        bodies_to_check = list(self.contained_bodies.values())
        
        while bodies_to_check: # We keep going until bodies_to_check is empty
            body_1 = bodies_to_check.pop()
            merge_occured = False
            
            # Iterate over the elements of the bodies to check
            # We don't use a for loop since we may delete a body at some point
            index = 0
            while index < len(bodies_to_check):
                body_2 = bodies_to_check[index]
                
                if self.have_bodies_collided(body_1, body_2):
                    merge_occured = True
                    body_1 = self.merge_bodies(body_1, body_2)
                    del bodies_to_check[index] # We remove body_2 from the list
                    index -= 1 # This changes the list's size so we decrement index
                
                index += 1 # Increment index to keep iterating through the list
                
            # Finally, we only add the merged body back to the list if it has merged
            # with something, meaning we still need to check it for more collisions
            if merge_occured:
                bodies_to_check.append(body_1)


    def barnes_hut_collision_handling(self) -> None:
        """This function merges bodies that have collided using the Barnes-Hut 
        quadtree. For each body, it does this by excluding clusters of other 
        bodies too far away to collide with that body (therefore speeding up 
        the collision detection) and otherwise checking for collisions with 
        other bodies (if they are close enough to collide)."""
        
        merge_occured = True
        while merge_occured: # The process is repeated until no merges happen
            merge_occured = False
            removed_bodies: set[Body] = set()
            self.merge_bodies_in_same_location()
            contained_bodies_list = list(self.contained_bodies.values())
            self.barnes_hut_tree = BarnesHutTree(contained_bodies_list)

            for body in contained_bodies_list:
                if body in removed_bodies: continue # We skip over removed bodies
                
                # We now perform a depth-first search of the quadtree using a stack
                stack = [self.barnes_hut_tree.root_node]
                while stack:
                    node = stack.pop()
                    
                    # Checks if the node's square contains only one body 
                    # (which is not body and has not been removed)
                    # If so we check for a collision between body and that body
                    if (node.num_contained_bodies == 1 
                        and (other_body := node.contained_bodies[0]) != body 
                        and other_body not in removed_bodies
                        and self.have_bodies_collided(body, other_body)):
                        
                        merge_occured = True
                        removed_bodies.add(body)
                        removed_bodies.add(other_body)
                        body = self.merge_bodies(body, other_body)
                         
                    elif node.num_contained_bodies > 1:
                        # Otherwise, we check if it's possible for body_1 to 
                        # collide with the bodies contained in the node's square
                        # and add its children to the stack if so
                        
                        displacement = body.position - node.center
                        cropped_vector = Vector(
                            max(0, abs(displacement.x_component) - node.width/2),
                            max(0, abs(displacement.y_component) - node.width/2))
                        minimum_distance = cropped_vector.magnitude   

                        if minimum_distance < body.radius+node.max_radius:
                            for child in node.children:
                                stack.append(child)
                                     
                                                
    def direct_sum_acceleration_calculation(self) -> None:
        """This function uses the direct sum method to find the exact 
        acceleration on each body. This means finding the gravitational 
        attraction between each pair of bodies, resolving it into acceleration 
        vectors and adding them to the acceleration of each body."""
                
        contained_bodies_list = list(self.contained_bodies.values())
                
        # A body's acceleration is replaced when it is calculated
        # Therefore we start by setting the acceleration of all bodies to 0
        for body in contained_bodies_list:
            body.acceleration = Vector(0, 0)

        # This nested loop iterates through each pair of bodies
        for i in range(len(self.contained_bodies)):
            body_1 = contained_bodies_list[i]
            for j in range(i+1, len(self.contained_bodies)):
                body_2 = contained_bodies_list[j]
                
                # We now find the acceleration vectors due to the 
                # gravitational attraction between the bodies on each body
                
                # We now find the softened displacement vectors and distance
                # between this pair of bodies.
                displacement_vector_1 = body_2.position - body_1.position
                displacement_vector_1.soften(self.epsilon)
                displacement_vector_2 = -displacement_vector_1
                softened_distance = displacement_vector_1.magnitude
                            
                # The magnitude of the acceleration due to gravity on each 
                # body is found using Newton's second law (a = F/m), Newton's 
                # law of gravitation and softening
                acceleration_magnitude_1 = self.G*body_2.mass/softened_distance**2
                acceleration_magnitude_2 = self.G*body_1.mass/softened_distance**2
                
                # The acceleration vectors due to gravity point from one body 
                # to another, so their directions are the same as the 
                # displacement vectors
                acceleration_1 = Vector(
                    magnitude = acceleration_magnitude_1,
                    direction = displacement_vector_1.direction)
                acceleration_2 = Vector(
                    magnitude = acceleration_magnitude_2,
                    direction = displacement_vector_2.direction)
        
                body_1.acceleration += acceleration_1
                body_2.acceleration += acceleration_2
    

    def barnes_hut_acceleration_calculation(self) -> None:
        """This function uses the Barnes-Hut algorithm to find the approximate
        acceleration on each body. It approximates because the acceleration of 
        a body towards a group of distant bodies is found by treating the 
        group as a point mass and using its center of mass and total mass 
        as if it was a single body."""
        
        self.merge_bodies_in_same_location()
        contained_bodies_list = list(self.contained_bodies.values())
        self.barnes_hut_tree = BarnesHutTree(contained_bodies_list)
        
        for body in self.contained_bodies.values():
            # A body's acceleration is replaced each time it is calculated, so 
            # at first we set it to zero
            body.acceleration = Vector(0, 0)
            
            # We now traverse the quadtree in a depth-first search using a stack
            stack = [self.barnes_hut_tree.root_node]
            while stack:
                node = stack.pop()
                                
                # We now find the softened distance between the body and the 
                # node's square's center of mass
                displacement_vector = node.center_of_mass - body.position
                displacement_vector.soften(self.epsilon)
                softened_dist = displacement_vector.magnitude
                
                # If the square's width divided by the distance is less than theta
                # it is considered far away, so its effect on the body will be
                # approximated by treating the bodies in the square as a single
                # body (using the center of mass and total mass of the bodies).
                # We also do this if it contains only 1 body, 
                # assuming we are not iterating over that body.
                
                is_far_away = softened_dist != 0 and node.width/softened_dist < self.theta
                contains_one_body = (node.num_contained_bodies == 1)
                if contains_one_body:
                    contained_body = node.contained_bodies[0]
                                
                if ((is_far_away or contains_one_body) 
                    and not (contains_one_body and contained_body == body)):
                    
                    # The approximated magnitude of the acceleration due to 
                    # gravity on the body due to the bodies in the node's square
                    acceleration_magnitude = self.G*node.total_mass/softened_dist**2
                    
                    # The direction of the acceleration vector is the same as 
                    # the direction of the displacement vector since the body 
                    # is attracted towards the node's center of mass
                    acceleration_on_body = Vector(
                        magnitude = acceleration_magnitude, 
                        direction = displacement_vector.direction)
                    
                    body.acceleration += acceleration_on_body 
                    
                # Otherwise this node's children (if any) are pushed to the stack
                elif node.num_contained_bodies > 1:
                    for child in node.children:
                        stack.append(child)
        

    def step_euler(self) -> None:
        """This function steps each body in the 
        simulation using the Euler method."""
        
        self.time_elapsed += self.time_step
        for body in self.contained_bodies.values():
            body.step_euler(self.time_step)


    def step(self) -> None:
        """This function combines acceleration calculation, collision 
        detection and the Euler method to step the simulation forward one 
        frame (i.e. it gets the next "snapshot" of the simulation in time)."""
        
        # Executes the collision detection algorithm chosen by the user
        match self.collision_detection:
            case "Barnes-Hut": 
                self.barnes_hut_collision_handling()
            case "Direct Sum": 
                self.direct_sum_collision_handling()
            case "Direct Sum (Optimized)": 
                self.optimized_direct_sum_collision_handling()
            case "None": 
                pass
            
        # Executes the acceleration calculation algorithm chosen by the user
        match self.acceleration_calculation:
            case "Barnes-Hut": 
                self.barnes_hut_acceleration_calculation()
            case "Direct Sum": 
                self.direct_sum_acceleration_calculation()
 
        # Moves all the bodies in the simulation using the Euler method
        self.step_euler()


    def add_body(self, mass: int|float, radius: int|float, position: Vector) -> Body:
        """This function is used to add a body to the simulation.
        The new body is returned for use in the PygameWindow class."""
        
        new_body = Body(self.find_unique_id(), mass, radius, position)
        self.contained_bodies[new_body.id] = new_body
        return new_body


    @property
    def center_of_mass(self) -> Body:
        """This function returns the center of mass of the simulation as a 
        Body object. This is used to set the centered object of the simulation
        to this body when selected."""
        
        center_of_mass = Vector(0, 0)
        total_mass = 0
        
        for body in self.contained_bodies.values():
            center_of_mass += body.position*body.mass
            total_mass += body.mass
            
        if total_mass != 0:
            center_of_mass /= total_mass
                      
        return Body("Center of mass", 0, 0, center_of_mass)


class PygameWindow(Simulation):
    """This class is responsible for drawing the grid, the bodies on it, etc. 
    It also checks for user events (e.g. clicking on a body)."""    
    
    def __init__(self) -> None:
        """This function initializes this object and its attributes along with
        the pygame library, causing the pygame window to appear to the user."""
                
        super().__init__() # Initializes the parent Simulation class
        pygame.init() # Initializes the pygame library
        
        # This block gets the window dimensions of the screen (in pixels) as a Vector
        screen_information = pygame.display.Info()
        self.window_dimensions = Vector(
            screen_information.current_w, 
            screen_information.current_h)
        
        # Sets the pygame window to be fullscreen and scaled to the user's monitor
        # self.pygame_screen is a pygame Surface object onto which we draw everything
        self.pygame_screen = pygame.display.set_mode(
            self.window_dimensions.components, 
            pygame.FULLSCREEN | pygame.SCALED)
        
        # This next block initializes all the attributes of this class
        
        self.clock = pygame.time.Clock()
        self.grid_square_length: int = 30 # Small grid square length in pixels
        self.zoom_factor = 1 # How zoomed in/out the grid is
        self.centered_body = Body("Origin", 0, 0, Vector(0, 0))
        self.frames_per_second = 60 # The number of simulation steps per second
        self.is_mouse_held = False # Is the left mouse button held down or not
        self.selected_body: Body|None = None
        self.is_simulation_paused = True
        
        # The lengths of vector arrows drawn are multiplied by this value
        self.arrow_length_factor = 1
        
        # The offset due to the user dragging the grid around
        self.grid_offset = Vector(0, 0)
        
        # If True, the user can click on the screen to add a body
        self.adding_body_mode = False
        
        # These two attributes store input from the orbit generation window
        self.body_to_orbit_around: Body|None = None
        self.semi_major_axis_length = 0
        
        # A trail is shown tracing the movements of the selected body if 
        # self.show_trail is True. trail_points stores the previous positions 
        # of the selected and centered body in order to draw the trail.
        self.show_trail = False
        self.trail_points: list[tuple[Vector, Vector]] = [] 
        
        
    def translate_coordinates(self, 
                              coordinates: Vector, 
                              mode: Literal["from", "to"], 
                              centering_offset: None|Vector = None) -> Vector:
        """This function either translates simulation coordinates TO pygame 
        coordinates or gets simulation coordinates FROM pygame coordinates"""
        
        # The flipping of the y component is due to Pygame having a flipped y axis
        # compared to our simulation. In Pygame, a higher y value means something is
        # further down the screen, unlike in a normal coordinate system.
        # Multipling by zoom_factor converts a length in meters to a length in pixels.
        
        if centering_offset is None:
            centering_offset = self.centered_body.position
        total_pixel_offset = self.grid_offset+self.window_dimensions//2
        
        if mode == "to":
            coordinates -= centering_offset
            coordinates *= self.zoom_factor
            coordinates.y_component *= -1
            coordinates += total_pixel_offset
        else:
            coordinates -= total_pixel_offset
            coordinates.y_component *= -1
            coordinates /= self.zoom_factor
            coordinates += centering_offset
            
        return coordinates
               
        
    def check_events(self) -> None:
        """This function checks for events like the user clicking on the 
        screen or dragging their mouse, and resolves them into actions. For 
        example, if a user clicks on a body then that body is selected."""
        
        for event in pygame.event.get(): # Each event is a pygame event object
            if event.type == pygame.MOUSEWHEEL: # The user has used the scroll wheel
                # In this case we change the zoom of the grid.
                # event.y is positive or negative based on if the user has
                # moved their scroll wheel up or down.
                
                zoom_multiplier = 1+event.y/10
                self.zoom_factor *= zoom_multiplier
                self.grid_offset *= zoom_multiplier
                
                # If the grid squares become too small or too large, then we
                # reset them to the other end of the size spectrum
                self.grid_square_length = int(self.grid_square_length*zoom_multiplier)
                if self.grid_square_length < 10: self.grid_square_length = 30
                elif self.grid_square_length > 30: self.grid_square_length = 10
                
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # The user has clicked their left mouse button
                self.is_mouse_held = True
                
                # We convert the user's click position to simulation coordinates
                mouse_position = Vector(*pygame.mouse.get_pos())
                sim_coords = self.translate_coordinates(mouse_position, "from")
                
                # If self.adding_body_mode is True we add a body where the user 
                # clicked. Otherwise, we check if they clicked on a body.
                if self.adding_body_mode:
                    new_body_mass = 10**14/self.zoom_factor**2
                    new_body_radius = 20/self.zoom_factor
                    new_body = self.add_body(new_body_mass, new_body_radius, sim_coords)
                    self.selected_body = new_body
                    self.adding_body_mode = False
                else:
                    for body in self.contained_bodies.values():
                        # We find distance between the click location and the body
                        distance = (sim_coords - body.position).magnitude
        
                        if distance < body.radius:
                            # The user has clicked on this body. We change the selected 
                            # body to what they selected and clear the trail points 
                            # if it changed.
                            if body != self.selected_body:
                                self.trail_points = []
                            self.selected_body = body
                        
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: 
                # The user has let go of their left mouse button
                self.is_mouse_held = False
                
            elif event.type == pygame.MOUSEMOTION and self.is_mouse_held: 
                # The user is dragging the grid
                # event.rel is a tuple describing how much the user dragged the grid by
                self.grid_offset += Vector(*event.rel)
                
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:  
                # The user has pressed the space bar so we un/pause the simulation
                self.is_simulation_paused = not self.is_simulation_paused


    def draw_all(self) -> None:
        """This function combines all of the draw_ functions to draw 
        everything onto the Pygame window."""
        
        # First we fill the screen with black
        self.pygame_screen.fill("black")
        
        # Next, we draw everything onto the screen. 
        # Note the order: if we were to call draw_grid() after draw_bodies() 
        # then grid lines would be displayed on top of bodies
        self.draw_grid()
        self.draw_bodies()
        self.draw_extra()


    def draw_grid(self) -> None:
        """This function draws the grid onto the pygame screen. It also draws 
        the grid indicator onto the screen, showing the size of a large grid 
        square in various space units."""
                
        # Draw the small and large grid squares. 
        # A large grid square contains a 5 by 5 square of small grid squares
        self.draw_grid_lines(self.grid_square_length, "#141414")
        self.draw_grid_lines(self.grid_square_length*5, "#444444")
            
        # Draw the center grid lines representing the origin
        total_offset = self.window_dimensions//2 + self.grid_offset
        pygame.draw.line( # Draw the vertical line
            self.pygame_screen, 
            "#BBBBBB",  
            (total_offset.x_component, 0), 
            (total_offset.x_component, self.window_dimensions.y_component))
        pygame.draw.line( # Draw the horizontal line
            self.pygame_screen, 
            "#BBBBBB", 
            (0, total_offset.y_component), 
            (self.window_dimensions.x_component, total_offset.y_component))
        
        # Draw the space indicator showing the size of a large grid square
        self.draw_space_indicator()


    def draw_grid_lines(self, square_length: int, color: str) -> None:
        """This function draws a set of horizontal and vertical lines forming 
        a grid using the given parameters."""
        
        screen_width, screen_height = self.window_dimensions.components

        # First we find the x and y coordinate of the first vertical and 
        # horizontal grid lines respectively
        total_offset = self.window_dimensions//2 + self.grid_offset
        grid_start_point = total_offset % square_length
        first_x, first_y = map(int, grid_start_point.components)

        # Draw the vertical lines at different x coordinates
        for x_coordinate in range(first_x, screen_width, square_length):
            start_coordinate = (x_coordinate, 0)
            end_coordinate = (x_coordinate, self.window_dimensions.y_component)
            pygame.draw.line(self.pygame_screen, color, 
                             start_coordinate, end_coordinate)
    
        # Draw the horizontal lines at different y coordinates
        for y_coordinate in range(first_y, screen_height, square_length):
            start_coordinate = (0, y_coordinate)
            end_coordinate = (self.window_dimensions.x_component, y_coordinate)
            pygame.draw.line(self.pygame_screen, color, 
                             start_coordinate, end_coordinate)


    def draw_space_indicator(self) -> None:
        """This function finds the bottom-right most large grid square and 
        draws a bar around it showing, in text, its length in meters, 
        kilometers, astronomical units and light seconds."""
        
        square_length = self.grid_square_length*5
        
        # First we find the (x, y) coordinates of the bottom right point of 
        # the bottom-right most large grid square
        total_offset = self.window_dimensions//2 + self.grid_offset
        grid_start_point = total_offset % square_length
        top_left = (self.window_dimensions-grid_start_point)%square_length
        bottom_right = self.window_dimensions - top_left
        x, y = bottom_right.components
        
        vertices = ( # These are the points that make up the space indicator
            (x, y), 
            (x, y-square_length//5), 
            (x-square_length, y-square_length//5), 
            (x-square_length, y))
        
        pygame.draw.lines(self.pygame_screen, "white", False, vertices)
        
        # Now we draw the font onto the space indicator, showing its size in 
        # various space units in 3 decimal place standard form.
        font = pygame.font.Font(size=30)
        square_length /= self.zoom_factor
        
        # f"{x:.3e}" means displaying x in 3 decimal place standard form and 
        # is used throughout this program
        texts_to_display = (
            font.render(f"{square_length:.3e} m", 0, "grey"), 
            font.render(f"{square_length/1000:.3e} km", 0, "grey"), 
            font.render(f"{square_length/149597870700:.3e} AU", 0, "grey"), 
            font.render(f"{square_length/299792458:.3e} Ls", 0, "grey"))
        
        text_widths = (text.get_width() for text in texts_to_display)
        max_width = max(text_widths)
        
        for index, text in enumerate(texts_to_display):
            text_coordinates = (
                x - max_width, 
                y - (square_length*self.zoom_factor)//5 - 25*(index+1))
            self.pygame_screen.blit(text, text_coordinates)
        
                
    def draw_bodies(self) -> None:
        """This function draws all the bodies in the simulation onto the 
        screen. It also draws any extra parts of bodies like vector arrows 
        drawn onto them, orbit previews, trail points or a preview of a body 
        being added to the simulation.""" 
        
        # Draw the bodies in the simulation onto the screen
        for body in self.contained_bodies.values():
            body_pygame_coords = self.translate_coordinates(body.position, "to")
            displayed_radius = body.radius*self.zoom_factor
            
            color = "blue"
            if body == self.selected_body:
                color = "yellow"
                
            pygame.draw.circle(self.pygame_screen, color, 
                               body_pygame_coords.components, displayed_radius)
        
        # We now add any extra items to the selected body if there is one.
        # This includes a trail, a preview of its orbit, and vector arrows.
        if self.selected_body and self.selected_body.id in self.contained_bodies:
            if self.show_trail and len(self.trail_points) >= 2:
                self.draw_trail_points()
            
            if self.semi_major_axis_length and self.body_to_orbit_around:
                self.draw_orbit_preview()
            
            if not self.selected_body.is_immobile:
                self.draw_arrow_on_selected_body(
                    "red", self.selected_body.velocity.copy())
                self.draw_arrow_on_selected_body(
                    "purple", self.selected_body.acceleration.copy())
                self.draw_arrow_on_selected_body(
                    "green", self.selected_body.engine_acceleration.copy())
            
        # Finally, the user will be shown a preview of the body they are
        # adding on the screen if adding_body_mode is True
        if self.adding_body_mode:
            pygame.draw.circle(self.pygame_screen, "blue", pygame.mouse.get_pos(), 20)
            
            
    def draw_arrow_on_selected_body(self, color: str, vector: Vector) -> None:
        """This function draws a vector arrow on the selected body."""
        
        if vector.magnitude == 0:
            return # If so we exit the function as the vector arrow will have no length
        else:
            # Refer back to translate_coordinates for an explanation
            vector.y_component *= -1
            vector *= self.arrow_length_factor * self.zoom_factor
        
        pygame_coords = self.translate_coordinates(self.selected_body.position, "to")
        arrow_tip_coordinates = pygame_coords+vector
        selected_body_pixel_radius = self.selected_body.radius*self.zoom_factor

        # Draw the stem of the arrow (i.e. the part without the arrowhead)
        stem_width = math.ceil(1/2*selected_body_pixel_radius) 
        pygame.draw.line(self.pygame_screen, color, pygame_coords.components,
                         arrow_tip_coordinates.components, stem_width)
        
        # Now we draw the arrowhead 
        vector.magnitude = selected_body_pixel_radius
        normal_vector = Vector(-vector.y_component, vector.x_component)
        
        arrowhead_vertices = (
            (arrow_tip_coordinates+normal_vector).components,
            (arrow_tip_coordinates-normal_vector).components,
            (arrow_tip_coordinates+vector).components)
        
        pygame.draw.polygon(self.pygame_screen, color, arrowhead_vertices)


    def draw_orbit_preview(self) -> None:
        """This function draws an orbit preview for the selected body
        if a user has inputted both a semi major axis length and a body to 
        orbit around. This preview depends on the assumptions made in the 
        equation for calculating a stable orbit (see section 1.3.3), so 
        if any of these are broken this preview may not be accurate."""
                    
        # The distance between the body to orbit around and the selected body
        displacement_vector = (self.body_to_orbit_around.position 
                               - self.selected_body.position)
        distance = displacement_vector.magnitude
        
        # If true then either the semi major axis length is too small or too 
        # large, so we do not draw a preview.
        if not distance/2 <= self.semi_major_axis_length <= distance*5:
            return
            
        ellipse_length = 2*self.semi_major_axis_length
        ellipse_width = 2*math.sqrt(distance*(ellipse_length-distance))
        ellipse_dimensions = Vector(ellipse_length, ellipse_width)
        ellipse_dimensions *= self.zoom_factor # Convert from meters to pixels
        
        # Now we create and rotate the ellipse
        ellipse_surface = pygame.Surface(ellipse_dimensions.components, pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surface, "green", ellipse_surface.get_rect(), 5)
        rotation_angle = math.degrees(displacement_vector.direction)
        ellipse_surface = pygame.transform.rotate(ellipse_surface, rotation_angle)
        
        # The dimensions of the smallest rectangle containing the ellipse
        bounding_box = Vector(*ellipse_surface.get_rect().size)
        
        # We now find the pygame coordinates of the top left of the ellipse
        displacement_vector.magnitude = self.semi_major_axis_length
        ellipse_center = self.selected_body.position + displacement_vector 
        pygame_center = self.translate_coordinates(ellipse_center, "to")
        top_left = pygame_center - bounding_box/2
        
        self.pygame_screen.blit(ellipse_surface, top_left.components)
                
    
    def draw_trail_points(self) -> None:
        # We now process self.trail_points into a list of tuples of 
        # coordinates to draw lines through
        drawn_points: list[tuple[int, int]] = []
        
        for past_position, centering_offset in self.trail_points:
            # Using the past position of the selected and centered body we 
            # transform the past position into pygame coordinates
            pygame_coordinates = self.translate_coordinates(
                past_position, "to", centering_offset)
            drawn_points.append(pygame_coordinates.components)
            
        # This draws many straight lines going through each of the points 
        pygame.draw.lines(self.pygame_screen, "white", False, drawn_points) 
                
                
    def draw_extra(self) -> None:
        """This function draws miscellaneous items onto the pygame window, 
        namely the paused icon and the fps counter."""
        
        # If the simulation is paused we will display a paused icon on the 
        # screen so the user knows the simulation is paused
        if self.is_simulation_paused:
            pygame.draw.rect(self.pygame_screen, "grey", (50, 50, 40, 150))
            pygame.draw.rect(self.pygame_screen, "grey", (125, 50, 40, 150))

        # Next, we draw an indicator showing the current fps (frames per second).
        # This is already viewable in the user interface, but the value shown there
        # is inaccurate if the simulation is under heavy load
        font = pygame.font.Font(size=30)
        
        try:
            text = font.render(f"{int(self.clock.get_fps())} FPS", 0, "grey")
        except OverflowError: # An fps too high will trigger an overflow error
            text = font.render(f"Extremely high FPS", 0, "grey")
        text_width = text.get_width()
        text_coordinates = (self.window_dimensions.x_component-30-text_width, 50)
        self.pygame_screen.blit(text, text_coordinates)
                        
                
    def back_end_main_loop(self) -> None:
        """This is the main loop of the back-end of the simulation.
        We step bodies and also perform a few actions that cannot be done in 
        check_events() (as they should be done only after each simulation step 
        but check_events() runs repeatedly)."""
        
        while True:
            if not self.is_simulation_paused:
                self.step()  
                
                if not self.show_trail or self.selected_body is None:
                    self.trail_points = []
                else:
                    current_positions = (self.selected_body.position, 
                                         self.centered_body.position)
                    self.trail_points.append(current_positions)     
                    
                if self.centered_body.id == "Center of mass":
                    self.centered_body = self.center_of_mass
                  
            # We now wait a bit of time to ensure a constant frame rate
            self.clock.tick(self.frames_per_second)
            
            
    def remove_selected_body(self) -> None:
        """A function that removes the selected body from 
        the simulation (the only way of removing bodies)."""
        
        self.contained_bodies.pop(self.selected_body.id)
        
        # If the selected body was the centered body, we reset the centered
        # body to the origin and modify the grid offset so the user is still
        # looking at the same simulation coordinates after the centering change
        if self.centered_body == self.selected_body: 
            self.grid_offset = -self.centered_body.position*self.zoom_factor
            self.grid_offset.y_component *= -1
            self.centered_body = Body("Origin", 0, 0, Vector(0, 0))
            
        self.selected_body = None


    def generate_stable_orbit(self) -> None:
        """This function generates a stable orbit velocity for the selected 
        body around the body to orbit around at the semi major axis length 
        specified. This uses the Vis Viva equation and the assumptions that 
        come with it."""
        
        # First we find the distance between the centers of the two bodies
        displacement_vector = (self.body_to_orbit_around.position 
                                - self.selected_body.position)
        distance = displacement_vector.magnitude
            
        # We now find the magnitude and direction of the selected body's 
        # stable orbit velocity vector using the Vis Viva equation
        velocity_mag = math.sqrt(self.G*self.body_to_orbit_around.mass
                                 *(2/distance-1/self.semi_major_axis_length))
        PI = 3.14159265358979
        velocity_direction = displacement_vector.direction + PI/2
        
        self.selected_body.velocity = Vector(magnitude = velocity_mag, 
                                             direction = velocity_direction)


class UserInterface(PygameWindow):
    """This class is responsible for the user interface window. 
    Here the user can view, modify and graph various simulation parameters, 
    along with generating stable orbits and reading a user manual."""
    
    BODY_ATTRIBUTES = set(( # A set of all body attributes
        "id", "is_immobile", "position.x_component", 
        "position.y_component", "mass", "radius",
        "velocity.direction", "velocity.magnitude", 
        "velocity.x_component", "velocity.y_component", 
        "acceleration.direction", "acceleration.magnitude",
        "acceleration.x_component", "angular_momentum", 
        "acceleration.y_component", "engine_acceleration.direction", 
        "engine_acceleration.magnitude", "angular_velocity", 
        "engine_acceleration.x_component", "kinetic_energy",
        "engine_acceleration.y_component", "momentum.magnitude", 
        "momentum.x_component", "momentum.y_component"))
    
    
    def __init__(self) -> None:
        super().__init__() # Initializes the parent PygameWindow class
        
        # self.main_window is the base user interface window on which all 
        # other widgets are added
        self.main_window = tkinter.Tk()
        self.main_window.title("User interface")
        
        # This makes the user interface window appear on top of other ones
        self.main_window.attributes("-topmost", True)
        
        # This exits the program when the user clicks the X button on the user interface
        self.main_window.protocol("WM_DELETE_WINDOW", lambda: os.kill(os.getpid(), 9))
        
        self.main_window.resizable(False, False) # Resizing the window is disabled

        # These are the space units used to display values in the user interface
        self.space_units: Literal["Meters", "Kilometers", "Light seconds", 
                                  "Astronomical units (AU)"] = "Meters"
        
        # StringVars are tkinter objects assigned to widgets, used to get and set the
        # value displayed in a widget. self.stringvars maps an attribute name to the
        # StringVar for the widget displaying that attribute (usually an Entry widget).
        self.stringvars: dict[str, StringVar] = {}
        
        # This maps names of dropdown menus to their tkinter Combobox widget
        # A combobox widget is the widget for a dropdown menu
        self.dropdown_menus: dict[str, Combobox] = {}
        
        # This attribute contains lists of information about the graphed parameter
        # for each graph in the graphs window. Each list contains, in order:
        #   - The tkinter Canvas widget the graph is drawn on
        #   - The stringvar for the textbox showing the current value of the parameter
        #   - The stringvar for the textbox showing the time range of the graph
        #   - The time range value (how far back in time data points are shown for)
        #   - A list of (x, y) tuples of data points that store data to graph
        #   - The name of the attribute being graphed
        self.graphs: list[list[Canvas, StringVar, StringVar, int, 
                               list[tuple[int, int]], str]] = []
                        
        self.construct_user_interface()
        
        # We now start the back end main loop in a separate thread and then 
        # start the front end main loop. This is explained in section 2.2.3.
        back_end_thread = threading.Thread(target=self.back_end_main_loop)
        back_end_thread.start()
        self.front_end_main_loop() 
        
        
    def construct_user_interface(self):
        """This function creates the window bar at the top of the user interface
        and combines the _window functions to add every widget to it."""
        
        # A notebook is the widget you see at the top of the user interface window.
        # It allows you to choose between various windows (e.g., the "Basic buttons" window). 
        # (almost) each window is constructed by a _window function which returns
        # a widget containing all of the widgets for that window.
        main_notebook = Notebook(self.main_window)
        
        # The selected body menu window is itself a notebook
        # which is constructed using two _window functions
        selected_body_notebook = Notebook(main_notebook)
        selected_body_notebook.add(
            self.selected_body_info_window(main_notebook), 
            text="Attributes")
        selected_body_notebook.add(
            self.orbit_generator_window(main_notebook),
            text="Stable orbit generator")
        
        # Finally, we add all of the windows to main_notebook
        text_and_window: tuple[str, Frame|Notebook] = (
            ("Basic buttons", self.buttons_window(main_notebook)),
            ("Simulation parameters", self.parameters_window(main_notebook)), 
            ("Selected body menu", selected_body_notebook),
            ("Graphs", self.graphs_window(main_notebook)),
            ("Manual", self.manual_window(main_notebook)))
        for name, widget in text_and_window:
            main_notebook.add(child = widget, text = name)
        
        # .pack() is one way of adding widgets to the screen, allowing you to 
        # add widgets one after the other. The side parameter allows you to 
        # choose which side of the screen you add the widget to
        main_notebook.pack()
                    
                    
    # These next _window functions return a widget containing all of the widgets
    # for a window of the user interface. parent is the widget onto which this 
    # window is drawn onto
                        
    def buttons_window(self, parent: tkinter.Widget) -> Frame:
        """This function constructs a Frame object containing the "Basic 
        buttons" section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        # This function inverts a boolean attribute of this class given an attribute name
        invert_attr = lambda attr: setattr(self, attr, not getattr(self, attr))
        
        # Tuples of button text and associated commands
        text_and_command: tuple[tuple[str, function]] = (
            ("Quit", lambda: os.kill(os.getpid(), 9)), 
            ("Un/Pause", lambda: invert_attr("is_simulation_paused")), 
            ("Step", self.step), 
            ("Add body", lambda: invert_attr("adding_body_mode")), 
            ("Delete selected body", self.remove_selected_body), 
            ("Un/show trail on selected body", lambda: invert_attr("show_trail")))
        
        for text, command in text_and_command:
            Button(master=container, text=text, command=command).pack()
                
        return container 


    def parameters_window(self, parent: tkinter.Widget) -> Frame:
        """This function constructs a Frame object containing the "Simulation 
        parameters" section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        textbox_labels = (
            "Time elapsed (s): ", "Value of θ (>= 0): ", "Value of ε (su, >= 0): ",
            "Time step (> 0): ","Arrow length factor (>= 0): ", "Frames per second (> 0): ")
        
        # This for loops adds the textboxes to this window
        for index, label in enumerate(textbox_labels):
            attribute_name = self.label_to_attribute_name(label)
            attribute_value = getattr(self, attribute_name)
            
            label = Label(container, text=label)
            
            # .grid() is a way of adding widgets in a row-column format
            # sticky="w" means that the widget starts on the left side of its row
            # padx(x, 0) adds x pixels of padding to the left of the widget
            label.grid(row=index, sticky="w")
            
            textbox_stringvar = StringVar(self.main_window, f"{attribute_value:.3e}")
            self.stringvars[attribute_name] = textbox_stringvar
            textbox = Entry(container, relief="solid", width=20, 
                            textvariable=textbox_stringvar)
            
            # Time elapsed is a non-editable property so we make its textbox read only
            if attribute_name == "time_elapsed":
                textbox.configure(state="readonly")
                
            # Textbox bindings perform a command when an event happens with a widget
            # <Return> means the enter button was clicked. 
            # <FocusOut> means the user clicked away from that widget.
            # <ButtonPress-1> means that widget was clicked with the left mouse button
            # <<ComboboxSelected>> means a dropdown menu option was clicked
            # <<TreeviewSelected>> means a treeview option was clicked (see manual window)
            # The event paramter is automatically provided by Tkinter
            textbox.bind("<Return>", 
                         lambda event, attr=attribute_name: self.process_input(attr))
            
            def focus_out_command(event, attr=attribute_name):
                attribute_value = getattr(self, attr)
                translated_value = self.translate_space_units(attribute_value, attr, "to")
                self.stringvars[attr].set(f"{translated_value:.3e}")
            textbox.bind("<FocusOut>", focus_out_command)
            
            # <widget>.winfo_reqwidth gets the width in pixels of <widget>
            textbox.grid(row=index, sticky="w", padx=(label.winfo_reqwidth(), 0))
            
            
        # This is a tuple of tuples providing information about the parameters displayed
        # using dropdown menus. It contains the label text and dropdown options.
        dropdown_menus_info: tuple[tuple[str, tuple[str]]] = (
            ("Centered body: ", ("Origin")),
            ("Space units (su): ", ("Meters", "Kilometers",
                                    "Light seconds", "Astronomical units (AU)")),
            ("Acceleration calculation: ", ("Direct Sum", "Barnes-Hut")),
            ("Collision detection: ", ("Direct Sum", "Direct Sum (Optimized)", 
                                       "Barnes-Hut", "None")))
        
        # This for loop adds the dropdown menus to this window
        for index, parameter_info in enumerate(dropdown_menus_info):
            label, dropdown_options = parameter_info
            attribute_name = self.label_to_attribute_name(label)
            
            # Adds the label text to the left of the dropdown menu
            label = Label(container, text=label)
            label.grid(row=index+len(textbox_labels), sticky="w")
            
            dropdown_stringvar = StringVar(self.main_window, "")
            self.stringvars[attribute_name] = dropdown_stringvar
            dropdown_menu = Combobox(container, width=20, values=dropdown_options, 
                                     state="readonly", textvariable=dropdown_stringvar)

            dropdown_menu.current(0)
            
            # Next we find the right <<ComboboxSelected>> command for this dropdown
            if index == 0: # If so we find the command for the centeered body dropdown
                self.dropdown_menus["centered_body"] = dropdown_menu
                command = lambda event: self.process_input("centered_body")
            else:
                def command(event: tkinter.Event, attribute=attribute_name):
                    selected_value = event.widget.get()
                    setattr(self, attribute, selected_value)
                    
                    # Next we update the stringvars for the "semi_major_axis_length"
                    # and "epsilon" attributes since they don't update automatically
                    for attr in ("semi_major_axis_length", "epsilon"):
                        attribute_value = getattr(self, attr)
                        translated_value = self.translate_space_units(attribute_value,
                                                                      attr, 
                                                                      "to")
                        self.stringvars[attr].set(f"{translated_value:.3e}")
                
            dropdown_menu.bind("<<ComboboxSelected>>", command)

            dropdown_menu.grid(row=index+len(textbox_labels), sticky="w", 
                               padx=(label.winfo_reqwidth(), 0))
                                
        return container
        
        
    def selected_body_info_window(self, parent: tkinter.Widget) -> Frame:
        """This function constructs a Frame object containing the "Attributes"
        part of the "Selected body menu" section of the simulation interface 
        and returns it"""
        
        container = Frame(parent)
            
        scrollbar, canvas, frame = self.scrollable_frame(container, 375, 400)
        
        ATTRIBUTE_LABELS = (
            "id: ", "is immobile?: ", "x coordinate (su): ", 
            "y coordinate (su): ", "mass (kg, >= 0): ", "radius (su, > 0): ", 
            "velocity direction (rad): ", 
            "velocity magnitude (su s⁻¹, >= 0): ", "x velocity (su s⁻¹): ", 
            "y velocity (su s⁻¹): ", "acceleration direction (rad): ",
            "acceleration magnitude (su s⁻², >= 0): ", 
            "x acceleration (su s⁻²): ", "y acceleration (su s⁻²): ", 
            "engine acceleration direction (rad): ", 
            "engine acceleration magnitude (su s⁻², >= 0): ", 
            "engine x acceleration (su s⁻²): ", 
            "engine y acceleration (su s⁻²): ", 
            "x momentum (kg su s⁻¹): ", "y momentum (kg su s⁻¹): ", 
            "momentum magnitude (kg su s⁻¹, >= 0): ", 
            "kinetic energy (kg su² s⁻², >= 0): ", 
            "angular velocity (rad s⁻¹): ", "angular momentum (kg su² s⁻¹): ")
                
        for index, attribute_label in enumerate(ATTRIBUTE_LABELS):
            attribute_name = self.label_to_attribute_name(attribute_label)
            
            label = Label(frame, text=attribute_label)
            label.grid(row=index+1, sticky="w")
            
            textbox_stringvar = StringVar(self.main_window, "-")
            self.stringvars[attribute_name] = textbox_stringvar
            textbox = Entry(frame, relief="solid", width=20, 
                            textvariable=textbox_stringvar)

            NON_EDITABLE_ATTRIBUTES = set((
                "angular_velocity", "angular_momentum", "acceleration.magnitude", 
                "acceleration.x_component", "acceleration.y_component", 
                "acceleration.direction"))
            if attribute_name in NON_EDITABLE_ATTRIBUTES:
                textbox.configure(state="readonly")
                
            # If the user clicks on a textbox the simulation pauses
            textbox.bind(
                "<ButtonPress-1>", lambda event: setattr(self, "is_simulation_paused", True))
            
            textbox.bind(
                "<Return>", lambda event, attr=attribute_name: self.process_input(attr))
            
            textbox.grid(row=index+1, sticky="w", padx=(label.winfo_reqwidth(), 0))
            
        scrollbar.pack(side="right", fill="y")
        canvas.pack()
    
        return container
                
                
    def orbit_generator_window(self, parent: tkinter.Widget) -> Frame:
        """This function constructs a Frame object containing the "Orbit 
        generator" part of the "Selected body menu" section of the simulation 
        interface and returns it"""
        
        container = Frame(parent)
        
        # First we create the "body to orbit around" label and dropdown menu
        Label(container, text="Body to orbit around: ").grid(row=1, sticky="w")
        dropdown_stringvar = StringVar(self.main_window, "")
        self.stringvars["body_to_orbit_around"] = dropdown_stringvar
        dropdown_menu = Combobox(container, width=14, state="readonly", 
                                 textvariable=dropdown_stringvar)
        self.dropdown_menus["body_to_orbit_around"] = dropdown_menu
        
        dropdown_menu.bind(
            "<<ComboboxSelected>>", 
            lambda event: self.process_input("body_to_orbit_around"))
        dropdown_menu.grid(row=1, sticky="w", padx=(120, 0))
        
        # Next we create the label and textbox for the semi major axis length
        label = Label(container, text="Semi-major axis length (su, >= 0): ")
        label.grid(row=2, sticky="w")
        
        textbox_stringvar = StringVar(self.main_window, "0")
        self.stringvars["semi_major_axis_length"] = textbox_stringvar
        textbox = Entry(container, width=11, textvariable=textbox_stringvar)
        textbox.bind("<Return>", 
                     lambda event: self.process_input("semi_major_axis_length"))
          
        def focus_out_command():
            translated_length = self.translate_space_units(
                self.semi_major_axis_length, 
                "semi_major_axis_length", 
                "to"
            )
            textbox_stringvar.set(f"{translated_length():.3e}")
            
        textbox.bind("<FocusOut>", focus_out_command)
        textbox.grid(row=2, sticky="w", padx=(185, 0))
        
        # Finally we create a button that, when clicked, generates the stable orbit
        button = Button(container, text="Generate stable orbit", 
                        command=self.generate_stable_orbit)
        button.grid(sticky="w", padx=(60, 0))

        return container        
        

    def graphs_window(self, parent: tkinter.Widget) -> Frame:
        """This function constructs a Frame object containing the "Graphs" 
        section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        scrollbar, frame_background, frame = self.scrollable_frame(container, 210, 400)
        
        # Contains all of the labels of parameters the user can choose to graph
        graph_attribute_options = (
            "nothing", "selected body x coordinate", 
            "selected body y coordinate", "selected body velocity direction", 
            "selected body velocity magnitude",  "selected body x velocity", 
            "selected body y velocity", 
            "selected body acceleration direction", 
            "selected body acceleration magnitude",
            "selected body x acceleration", "selected body y acceleration",
            "selected body momentum magnitude", "selected body x momentum",
            "selected body y momentum", "selected body kinetic energy",
            "selected body angular velocity" , 
            "selected body angular momentum",
            "total x velocity", "total y velocity", 
            "total velocity magnitude", "total x acceleration",
            "total y acceleration", "total acceleration magnitude",
            "total x momentum", "total y momentum",
            "total momentum magnitude", "total kinetic energy",
            "total angular velocity", "total angular momentum")
        
        for i in range(10): # We now add 10 individual graph boxes
            graph_container = Frame(frame)
            
            # We now create the graph canvas and draw the graph axes and
            # labels, which are permanent
            canvas = Canvas(graph_container, width=200, height=150, background="white")
            
            # Draw the x axis and its label
            canvas.create_line(25, 130, 180, 130, fill="black")
            canvas.create_text(100, 140, text="Real time (s)")
            
            # Draw the y axis and its label
            canvas.create_line(25, 20, 25, 130, fill="black")
            canvas.create_text(10, 75, text="Parameter value", angle=90)
            
            canvas.grid(sticky="w", row=0)
            
            # Next we add the dropdown menu to the graph box for the selected parameter
            label = Label(graph_container, text="Selected parameter: ")
            label.grid(row=1, column=0, sticky="w")
            
            dropdown_menu = Combobox(graph_container, values=graph_attribute_options, 
                                     width=11, state="readonly")
            dropdown_menu.current(0)
            
            # This changes the width of the selection menu for the dropdown 
            # menu to be wide enough to fit all the names of the options
            Style().configure("TCombobox", postoffset=(0, 0, 150, 0))
            
            # When an option is selected in the dropdown, we modify self.graphs to clear
            # this graph's data points and set the attribute name to the seelcted one
            def on_option_select(event, graph_index=i):
                selected_attribute_name = self.label_to_attribute_name(event.widget.get())
                original_info_list = self.graphs[graph_index]
                new_info_list = original_info_list[:4] + [[], selected_attribute_name]
                self.graphs[graph_index] = new_info_list
            dropdown_menu.bind("<<ComboboxSelected>>", on_option_select)
            
            dropdown_menu.grid(row=1, sticky="w", padx=(label.winfo_reqwidth(), 0))
            
            # We now add the textbox showing the current value of the graphed parameter
            Label(graph_container, text="Current value: ").grid(row=2, sticky="w")
            
            current_value_stringvar = StringVar(graph_container, "-")
            current_value_textbox = Entry(graph_container, width=20, state="disabled",
                                          textvariable=current_value_stringvar)
            current_value_textbox.grid(row=2, sticky="w", padx=(80, 0))
            
            # We now add the textbox for the time range of the graph
            Label(graph_container, text="Time range (s, > 0): ").grid(row=3, sticky="w")
            time_range_stringvar = StringVar(graph_container, f"{5:.3e}")
            current_value_textbox = Entry(graph_container, width=15, 
                                          textvariable=time_range_stringvar)
            current_value_textbox.bind(
                "<Return>", 
                lambda event, index=i: self.process_input("time_range", index))
            
            def focus_out_command(event, index=i, stringvar=time_range_stringvar):
                time_range_value = f"{self.graphs[index][3]:.3e}"
                stringvar.set(time_range_value)
            current_value_textbox.bind("<FocusOut>", focus_out_command)
    
            current_value_textbox.grid(row=3, sticky="w", padx=(110, 0))
            
            graph_container.pack()
            
            # Refer back to line 1463 for an explanation of what self.graphs contains
            self.graphs.append([canvas, current_value_stringvar, 
                                time_range_stringvar, 5, [], "nothing"])
                        
        scrollbar.pack(side="right", fill="y")
        frame_background.pack()
                
        return container
    
    
    def manual_window(self, parent: tkinter.Widget) -> Frame:
        """This function constructs a Frame object containing the "Manual" 
        section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        # text_mapping contains a mapping that converts topic names to text describing them.
        # Topics that are names for groups of other topics have no description ("").
        text_mapping: dict[str, str] = {
            "How to use the interface": 
                "", 
            "Using textboxes": 
                "To input a value, simply click on the textbox and type your "
                "desired value into it. To the left of a textbox is in brackets, "
                "the unit of the paramete and any conditions (e.g. >= 0). "
                "For numerical inputs, 3 decimalm place standard form is used. "
                "ae+b means a × 10ᵇ and ae-b means a × 10⁻ᵇ. When you press enter, "
                "your input is processed. If it is invalid, it will be cleared. "
                "Otherwise, you will see the text cursor disappear. "
                "When you click away from a textbox, its value will be reset "
                "to the attribute it is displaying. Some textboxes are grayed out, "
                "meaning their values cannot be edited.",
            "Adding a body": 
                "To add a body, click on the 'Add body' button in the basic "
                "buttons section on the top-left. Then click on the screen where "
                "you want to add your body. You will be shown a preview. Once "
                "your body has been added, it will automatically be selected.",
            "Orbit generation": 
                "To generate an orbit, first select a body (it will be the "
                "orbiting body). Then select a body to orbit around and input "
                "a semi major axis length. The major axis length is the length "
                "of the longest line that can be drawn in the ellipse, and the "
                "semi-major axis length is half that. Next, you will be shown an "
                "orbit preview. If your input for the axis length is too short "
                "or too long, no preview will be displayed. Press the button to "
                "generate the orbit. This is ignored if the input for the axis "
                "length is too small.",
            "Using the graphs": 
                "Graphs have arbitrary axes scaling for the y axis (the axes "
                "just expand to graph all the points). The time range is the "
                "amount of time in the past for which the parameter is plotted. "
                "If you see 'small change warning', this means there has been "
                "only a small change in the value of the parameter you are  "
                "graphing. This could mean you are viewing the graph of a  "
                "conserved property which should not change but is shown to "
                "change because of floating point errors.",
            "Selecting a body": 
                "To select a body, simple click on it. It will be "
                "highlighted in yellow. Its velocity (red), acceleration (purple) "
                "and engine acceleration (green) will be shown as arrows.", 
            "How the simulation works": 
                "", 
            "Acceleration calculation": 
                "Acceleration calculation is either done using the Barnes-Hut "
                "algorithm or Direct Sum algorithm. The Direct Sum algorithm "
                "calculates the EXACT total acceleration on each body by going "
                "through each pair of bodies and calculating the force due to "
                "gravity between them. Note that this takes exponentially (squared) "
                "more time as the number of bodies increases. As an alternative, "
                "the Barnes-Hut algorithm APPROXIMATES the acceleration on each "
                "body. Let's say it's finding the acceleration on a body X. "
                "This algorithm clusters bodies far away from X and approximates"
                "the acceleration on X due to those bodies using the center of "
                "mass and total mass of the cluster. This is MUCH faster than the "
                "Direct Sum approach for a large number of bodies. Note that the "
                "Barnes-Hut method for acceleration calculation DOES NOT conserve "
                "momentum.",
            "Collision detection": 
                "Collision detection can either be disabled, Direct Sum (normal "
                "or optimized), or Barnes-Hut. Please read the acceleration "
                "calculation page for an explanation of Barnes-Hut and Direct-Sum. "
                "When applied to collision detection, Direct Sum simply checks for "
                "a collision between each pair of bodies. Meanwhile, Barnes-Hut is "
                "much faster as it excludes bodies that are too far away for "
                "collision. When finding the collisions on a body. The Direct Sum "
                "collision detection method still grows in time exponentially "
                "(even when using the optimized version), and the Barnes-Hut version "
                "is again much faster.",
            "How bodies move": 
                "Bodies are moved using the 'Euler method'. Each step or frame "
                "in the simulation, a body's velocity is incremented by its "
                "acceleration multiplied by the 'Time step' parameter. Then, its "
                "position is incremented by its velocity again multiplied by the "
                "time step. In more mathematical terms, 1. velocity = velocity + "
                "(acceleration due to gravity + engine acceleration) × time step. "
                "2. position = position + velocity × time step. A smaller time step "
                "value means a more accurate but slower simulation. "
                "The opposite is also true.",
            "Simulation parameters": 
                "",
            "The θ parameter": 
                "θ is the parameter used for the Barnes-Hut algorithm (acceleration "
                "calculation only). Please read the 'Acceleration calculation' section "
                "of 'How does the simulation work' first. A higher value of θ means "
                "that the acceleration on each body is approximated more, by "
                "decreasing the distance at which clustering happens (clustering "
                "is more accurate further away). This means a faster but less accurate "
                "simulation. The opposite is also true.",
            "The ε parameter": 
                "ε is the parameter used in a technique called 'smoothening'. "
                "Sometimes two bodies get too close and their acceleration spikes "
                "unnaturally. A higher value of ε reduces this effect, by setting "
                "the minimum distance we consider two bodies to have.",
            "Time step": 
                "Please see the 'How bodies move' section in the "
                "'How the simulation works' section.",
            "Arrow length factor": 
                "The length of vector arrows drawn on the selected "
                "body are multiplied by this value.",
            "Frames per second": 
                "This is how many times per second the Euler method is used. "
                "Note that this is not accurate if the simulation is under "
                "heavy load (so use the FPS counter). See the 'How bodies move' "
                "section in the 'How the simulation works' section for what the "
                "Euler method is.",
            "Centered object": 
                "This is the object placed at the center of the simulation. "
                "This can be either the origin, the center of mass of the "
                "simulation or a body.",
            "Space units (su)": 
                "This changes the units for space used in the simulation. "
                "Properties like coordinates, velocities and so on are scaled to "
                "be in the right units. For example, if the space unit selected "
                "is light seconds, then coordinates will be in light seconds.",
            "Selected body id": 
                "This is a unique identifier given to every body. This means that "
                "two id's cannot be the same. An id is also not allowed to be either "
                "'Center of mass' or 'Origin'",
            "Angular properties": 
                "Angular properties (angular velocity or momentum) are calculated "
                "using the centered object of the simulation. Angular velocity is "
                "calculated by finding the change in angle from the centered body "
                "to the selected body divided by the time step. Angular momentum is "
                "calculated using the cross product of the vector from the centered "
                "body to the selected body and the selected body's velocity."
        }
                
        # We now add newlines (\n) to limit the maximum length of the 
        # explanation on the screen
        maximum_length = 36 # Maximum length of a line in characters
        for topic, explanation in text_mapping.items():
            modified_explanation = ""
            words = explanation.split(" ")
            
            current_line = ""
            for word in words:
                if len(current_line+word) > maximum_length:
                    # We have reached our character limit and make a new line
                    modified_explanation += current_line + "\n"
                    current_line = word + " "
                else: # Otherwise we simply add the word to the current line
                    current_line += word + " "
            modified_explanation += current_line + "\n"
                    
            text_mapping[topic] = modified_explanation
                
        # This stringvar stores the current text displayed by the manual
        current_text_displayed = StringVar(container, "")
        text_widget = Label(container, justify="left", # Justify=left left aligns the text
                            textvariable=current_text_displayed)
        
        # This widget will be used to show a hierarchical view of the manual topics
        navigation_interface = Treeview(container, height=20, show="tree")
        
        # This contains tuples of tuples, each containing first a category 
        # name and then a parent. Each category name is one of the topic names 
        # from the text_mapping dictionary above. Parent is the id of the 
        # topic that that topic is under. For example, the "Using textboxes" 
        # category has parent "How to use the interface". Parent is left blank 
        # ("") if that topic has no parent
        CATEGORIES_INFO: tuple[tuple[str, str]] = (
            ("How to use the interface", ""), 
            ("Using textboxes", "How to use the interface"),
            ("Adding a body", "How to use the interface"),
            ("Orbit generation", "How to use the interface"),
            ("Using the graphs", "How to use the interface"),
            ("Selecting a body", "How to use the interface"),
            ("How the simulation works", ""),
            ("Acceleration calculation", "How the simulation works"),
            ("Collision detection", "How the simulation works"),
            ("How bodies move", "How the simulation works"),
            ("Simulation parameters", ""),
            ("The θ parameter", "Simulation parameters"),
            ("The ε parameter", "Simulation parameters"),
            ("Time step", "Simulation parameters"),
            ("Arrow length factor", "Simulation parameters"),
            ("Frames per second", "Simulation parameters"),
            ("Centered object", "Simulation parameters"),
            ("Space units (su)", "Simulation parameters"),
            ("Selected body id", "Simulation parameters"),
            ("Angular properties", "Simulation parameters"))
        
        for index, (id, parent) in enumerate(CATEGORIES_INFO):
            navigation_interface.insert(parent, index, id, text=id)

        # When a topic is selected the label's contents will be updated
        # to contain the explanation for that topic using its stringvar
        def on_select(event):
            selected_topic = navigation_interface.focus()
            topic_description = text_mapping[selected_topic]
            current_text_displayed.set(topic_description)
        navigation_interface.bind("<<TreeviewSelect>>", on_select)
        
        navigation_interface.pack(side="left")
        text_widget.pack(side="left")
                
        return container


    def scrollable_frame(self, 
                         parent: tkinter.Widget, 
                         width: int,
                         height: int) -> tuple[Scrollbar, Canvas, Frame]:
        """This function returns scrollbar, canvas and frame widgets that 
        are combined to allow for a scrollable widget in tkinter"""
        
        container = Canvas(parent, width=width, height=height)
        scrollable_frame = Frame(container)
        
        # container_background.yview_moveto adjusts container_background so
        # that a fraction of its height is off-screen at the top.
        # As the scrollbar goes down the fraction increases.
        scrollbar = Scrollbar(parent, orient="vertical", command=container.yview)
        
        # When a widget is added to scrollable_frame it increases in size to fit it.
        # This is performed by the lambda function shown below
        scrollable_frame.bind(
            "<Configure>", 
            lambda event: container.configure(scrollregion=container.bbox("all")))
        
        container.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # This makes the scrollbar's height automatically adjusts to the size of the frame
        container.configure(yscrollcommand=scrollbar.set) 
        
        return scrollbar, container, scrollable_frame


    def label_to_attribute_name(self, label_text: str) -> str:
        """This function converts the label text for a parameter to its 
        attribute name (e.g. "x velocity (su s⁻¹): " is converted to "velocity.
        x_component)."""
        
        # This next block converts the label text to snake case 
        # e.g. "x velocity (su s⁻¹): " would be converted to "x_velocity"
             
        label_text = label_text.replace("selected body ", "")
        label_text = label_text.replace("?", "")
        label_text = label_text.replace(": ", "")
        
        if "(" in label_text: # Removes brackets indicating attribute units
            bracket_index = label_text.index(" (")
            label_text = label_text[:bracket_index]
                        
        label_text = label_text.lower()
        snake_case = label_text.replace(" ", "_")
        
        # This next block performs miscellaneous modifications
        
        # Used to check if a string contains any of the given substrings
        def contains_any(string: str, substrings: list[str]) -> bool:
            return any([substring in string for substring in substrings])
        
        # This checks if the snake case is referencing a vector
        def is_vector(snake_case):
            vector_names = ["momentum", "velocity", "coordinate", "acceleration"]
            has_vector_name = contains_any(snake_case, vector_names)
            strings_to_exclude = ["angular", "calculation"]
            contains_excluded = contains_any(snake_case, strings_to_exclude)
            if has_vector_name and not contains_excluded:
                return True
            else:
                return False
        
        if snake_case == "value_of_θ": attribute_name = "theta"
        elif snake_case == "value_of_ε": attribute_name = "epsilon"
        elif is_vector(snake_case):
            if "engine_" in snake_case: prefix = "engine_"
            elif "total_" in snake_case: prefix = "total_"
            else: prefix = ""
            
            snake_case = snake_case.replace(prefix, "")
            snake_case = snake_case.replace("coordinate", "position")
            
            # If true we need to transform this snake_case from "x_<vector>" 
            # to "<vector>_x_component"
            if snake_case[0] in ("x", "y"):
                snake_case = f"{snake_case[2:]}_{snake_case[0]}_component"
            
            vector_name, vector_property = snake_case.split("_", maxsplit=1)
            attribute_name = prefix + vector_name + "." + vector_property
        else:
            attribute_name = snake_case
        
        return attribute_name      


    def translate_space_units(self, 
                              value: int|float, 
                              attribute_name: str,
                              mode: Literal["from", "to"]) -> int|float:
        """This function translates a value for a certain attribute between 
        different space units. Some attributes simply do not change if their 
        units do not depend on space. You specify if you want to convert FROM 
        space units to meters or from meters TO space units."""
        
        # A set of all attributes with units that include space units
        ATTRIBUTES_WITH_SPACE = set((
            "semi_major_axis_length", "position.x_component", 
            "position.y_component", "radius", "velocity.magnitude", 
            "velocity.x_component", "velocity.y_component", 
            "acceleration.magnitude", "acceleration.x_component", 
            "acceleration.y_component", "engine_acceleration.magnitude", 
            "engine_acceleration.x_component", "kinetic_energy",
            "engine_acceleration.y_component", "momentum.magnitude", 
            "momentum.x_component", "momentum.y_component", 
            "angular_momentum", "epsilon"))
        
        if attribute_name in ATTRIBUTES_WITH_SPACE:
            match self.space_units:
                case "Kilometers": conversion_scalar = 1000
                case "Astronomical units (AU)": conversion_scalar = 149597870700
                case "Light seconds": conversion_scalar = 299792458
                case "Meters": conversion_scalar = 1
                
            match mode:
                case "from": order = 1 # This means we multiply by the scalar
                case "to": order = -1 # This means we divide by the scalar
                
            # These attributes have a factor of su² so order of conversion is doubled
            if "kinetic_energy" in attribute_name or "angular_momentum" in attribute_name:
                order *= 2
                
            # We need to account for centering offsets in coordinates
            elif attribute_name == "position.x_component":
                value += order*self.centered_body.position.x_component
            elif attribute_name == "position.y_component":
                value += order*self.centered_body.position.y_component
                
            value *= conversion_scalar**order
        
        return value


    def process_input(self,
                      attribute_name: str|None,
                      graph_index: int|None = None) -> None:
        """This function checks a user's input for a certain attribute, and 
        sets the attribute to their inputted value if their input was valid. 
        It also updates the displayed value for that attribute."""
        
        # graph_index not being None or attribute_name being None means we are 
        # updating the time range value at self.graphs[graph_index].
        # This also applies for self.check_valid_input
            
        is_valid_input, input = self.check_valid_input(attribute_name, graph_index)
        
        # This gets the name of the object whose attribute we need to update
        object_name = "self"
        if attribute_name in self.BODY_ATTRIBUTES:
            object_name = "self.selected_body"
                    
        if is_valid_input: # If the input is valid we update the value of the attribute
            # First we remove the focus from the widget to show we updated the value
            self.main_window.focus_set()
            
            # Now we set the atribute to the inputted value since the input was valid
            if attribute_name == "time_range":
                # This sets the time range value at self.graphs[index]
                self.graphs[graph_index][3] = input 
            else:
                if attribute_name == "id":
                    # The selected body's id changes so we update self.contained_bodies
                    self.contained_bodies.pop(self.selected_body.id)
                    self.contained_bodies[input] = self.selected_body
                    
                # Next, there is no setter for momentum, so we have to manually
                # set the momentum if the user is updating it
                if "momentum" in attribute_name:
                    attribute_name = attribute_name.replace("momentum", "velocity")
                    selected_body_mass = self.selected_body.mass
                    exec(f"self.selected_body.{attribute_name} = input/selected_body_mass")
                else:
                    exec(f"{object_name}.{attribute_name} = input")        
                                   
        # We now set the value displayed in this attribute's textbox to the actual value.
        # This will display the user's input being cleared if it was invalid.
        
        non_textbox_attributes = ("body_to_orbit_around","centered_body")
        if attribute_name not in non_textbox_attributes and attribute_name != "time_range":
            value_to_display = eval(f"{object_name}.{attribute_name}")
            value_to_display = self.translate_space_units(value_to_display, 
                                                          attribute_name, 
                                                          "to")
            if attribute_name not in ("is_immobile", "id"):
                value_to_display = f"{value_to_display:.3e}"
            self.stringvars[attribute_name].set(str(value_to_display))
        elif attribute_name == "time_range":
            # We set the time range stringvar at self.graphs[graph_index]
            time_range = f"{self.graphs[graph_index][3]:.3e}"
            self.graphs[graph_index][2].set(time_range)
                 
         
    def check_valid_input(self,
                          attribute_name: str, 
                          graph_index: int|None) -> tuple[bool, str|int|float]:
        """This function retrieves the input to an attribute, and returns 
        whether the input was valid or not. It also returns the processed input."""
        
        if attribute_name == "time_range":
            inputted_value = self.graphs[graph_index][2].get()
        else:
            inputted_value = self.stringvars[attribute_name].get()
        
        # GTEZ is attributes that should be greater than or equal to zero
        # GTZ is attributes that should be greater than zero
        GTEZ = set(("theta", "epsilon", "arrow_length_factor", "mass", 
                    "kinetic_energy", "semi_major_axis_length"))
        GTZ = set(("time_step", "frames_per_second", "radius", "time_range"))
                  
        valid_input = True # We start from the assumption that input is valid
        
        if attribute_name == "is_immobile":
            match inputted_value.lower():
                case "true": processed_value = True
                case "false": processed_value = False
                case _: valid_input = False
                
        elif attribute_name == "id":
            lowercase_ids = (id.lower() for id in self.contained_bodies)
            existing_ids = ("origin", "center of mass", *lowercase_ids)
            valid_input = not (inputted_value.lower() in existing_ids)
            valid_input &= inputted_value != "" # The id must be non-empty
            processed_value = inputted_value
            
        elif attribute_name in ("body_to_orbit_around", "centered_body"):
            # In this case we process the input to a Body object
            match inputted_value:
                case "Origin":
                    body = Body("Origin", 0, 0, Vector(0, 0))
                case "Center of mass":
                    body = self.center_of_mass
                case _ if inputted_value in self.contained_bodies:
                    body = self.contained_bodies[inputted_value]
                case _:
                    body = None
            processed_value = body
                                        
            # If the centered body has changed then we clear the trail points
            if attribute_name == "centered_body" and body.id != self.centered_body.id:
                self.trail_points = []
                                
        else: # Otherwise the input is simply numerical
            try:
                float_value = float(inputted_value) # May raise a ValueError
                processed_value = self.translate_space_units(float_value, 
                                                             attribute_name, 
                                                             "from")
                valid_input = not((attribute_name in GTEZ and processed_value < 0) 
                                  or (attribute_name in GTZ and processed_value <= 0))
            except ValueError:
                # The input is invalid since it could not be converted to a float
                valid_input = False
                processed_value = ""
                    
        return valid_input, processed_value
         
         
    def set_widgets(self) -> None:
        """This function combines the three set_ functions to form a larger 
        function that sets every widget in the user interface."""
        
        self.set_dropdown_menus()
        self.set_selected_body_textboxes()
        self.set_graphs()
        
        self.stringvars["time_elapsed"].set(f"{self.time_elapsed:.3e}")
         

    def set_dropdown_menus(self) -> None:
        """This code refreshes the two dropdown menus in the user interface 
        (the one for the centered body and the body to orbit around).
        It sets the values that can be selected and also performs any checks needed."""
        
        # Find and set the dropdown options for the centered body dropdown menu
        dropdown_options = ["Origin"]
        dropdown_menu = self.dropdown_menus["centered_body"]
        if len(self.contained_bodies) > 0:
            dropdown_options.append("Center of mass")
            body_ids = tuple(self.contained_bodies)
            dropdown_options.extend(body_ids)
            
        # If the centered body does not exist (e.g. if it was 
        # deleted by the user), we set the centered body to the origin.
        if dropdown_menu.get() not in dropdown_options:
            dropdown_menu.current(0) # Sets the selected option to the first option
            self.centered_body = Body("Origin", 0, 0, Vector(0, 0))
        
        dropdown_menu.configure(values=dropdown_options)

        # Next we find and set the dropdown options for the body to orbit around 
        dropdown_menu = self.dropdown_menus["body_to_orbit_around"]   
        dropdown_options = []  
        if self.selected_body is None: # If so then there are no dropdown options
            dropdown_menu.set("") # Sets the selected value of the dropdown menu to ""
        else: # Otherwise the options are all body ids except for the selected body id
            for id in self.contained_bodies:
                if id != self.selected_body.id:
                    dropdown_options.append(id)
                    
            if dropdown_menu.get() not in dropdown_options:
                self.body_to_orbit_around = None
                dropdown_menu.set("")
                    
        dropdown_menu.configure(values=dropdown_options)


    def set_selected_body_textboxes(self) -> None:
        """This function sets the stringvar objects responsible for displaying 
        the properties of a body in the selected body menu attributes window."""
        
        for attribute_name in self.BODY_ATTRIBUTES:
            
            # If the user is currently inputting a value for this attribute, 
            # we do not set the stringvar (as this would clear their input)
            try:
                focussed_widget = self.main_window.focus_displayof()
                if isinstance(focussed_widget, Entry): # The Entry widget is a textbox
                    focussed_stringvar = focussed_widget.cget("textvariable").__str__()
                    attribute_stringvar = self.stringvars[attribute_name].__str__()
                    if attribute_stringvar == focussed_stringvar:
                        continue
            except KeyError: # A bug in tkinter sometimes happens which causes a KeyError
                pass
            
            # Otherwise, we set the stringvar for that attribute to the 
            # value of the attribute it should be displaying
            if self.selected_body is None:
                value = "-"
            else:
                if attribute_name == "angular_velocity":
                    value = self.selected_body.angular_velocity(self.centered_body, 
                                                                self.time_step)
                elif attribute_name == "angular_momentum":
                    value = self.selected_body.angular_momentum(self.centered_body)
                else:
                    value = eval(f"self.selected_body.{attribute_name}")
                
                value = self.translate_space_units(value, attribute_name, "to")
                        
                if attribute_name not in ("id", "is_immobile"):
                    value = f"{value:.3e}"
                    
            self.stringvars[attribute_name].set(str(value))


    def set_graphs(self) -> None:
        """This function sets the graphs widgets; this means updating the 
        values shown in the current value textbox, setting the data points
        that need to be drawn, and drawing the graph."""
        
        for index, graph_info in enumerate(self.graphs):
            (canvas, current_value_stringvar, time_range_stringvar, 
            time_range, data_points, attribute_name) = graph_info
                 
            # Checks if the attribute to be plotted is set to nothing or if a 
            # selected body property is selected but there is no selected body.
            # If so we clear the data points and display no current value.
            if (attribute_name == "nothing" 
                or "total" not in attribute_name and self.selected_body is None):
                
                current_value_stringvar.set("-")
                data_points = []
            else: 
                # Otherwise, we:
                #   - Set the value shown inside the current value textbox
                #   - Add a data point if the simulation is not paused
                #   - Remove data points too far in the past
                
                # First we set the value shown inside the current value textbox
                if "total" in attribute_name:
                    attribute_name = attribute_name.replace("total_", "")
                    displayed_value = self.total_of_property(attribute_name, 
                                                             self.centered_body)
                elif attribute_name == "angular_velocity":
                    displayed_value = self.selected_body.angular_velocity(self.centered_body,
                                                                          self.time_step)
                elif attribute_name == "angular_momentum":
                    displayed_value = self.selected_body.angular_momentum(self.centered_body)
                else:
                    displayed_value = eval(f"self.selected_body.{attribute_name}")
                    
                displayed_value = self.translate_space_units(displayed_value, 
                                                             attribute_name, 
                                                             "to")
                current_value_stringvar.set(f"{displayed_value:.3e}")
                
                # Now we add a data point and remove any points too far in the past
                if not self.is_simulation_paused:
                    # We add a data point if simulation time has passed or if there are none
                    if len(data_points) == 0 or data_points[-1][0] != self.time_elapsed:
                        data_points.append((self.time_elapsed, displayed_value))
                        
                    # Remove data points that are too far in the past.
                    frame_range = self.frames_per_second*time_range
                    number_to_remove = max(0, int(len(data_points)-frame_range))
                    del data_points[:number_to_remove]
                
            canvas.delete("data_points") # Clear the previously drawn data points
            canvas.delete("warning") # Delete the floating point warning text (see line 2495)
            if len(data_points) >= 2:
                self.draw_graph(canvas, data_points)      
                
            # Set the data points stored in self.graphs to the newly updated ones
            self.graphs[index][4] = data_points
                            
            
    def draw_graph(self, canvas: Canvas, data_points) -> None:
        """This function draws a graph on a given tkinter 
        Canvas object for the given data points."""
            
        x_values = [x for x, y in data_points]
        y_values = [y for x, y in data_points]
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        # We use these values to calculate scale factors and offsets used to scale the
        # data points to fit inside of the 100x150 pixel area we draw them inside of
        width_scaling = 150/(max_x-min_x)
        width_offset = 25-min_x*width_scaling
        height_scaling = 100/((max_y-min_y)+(max_y-min_y==0))
        height_offset = 25-min_y*height_scaling
        
        # If there is a very small change in the value of the parameter, we 
        # warn the user there may be a floating point error.
        last_y = f"{data_points[-1][1]:.3e}"
        second_to_last_y = f"{data_points[-2][1]:.3e}"
        if last_y == second_to_last_y:
            canvas.create_text(100, 10, text="Small change warning", 
                               tags="warning", fill="red")
        
        # We now transform the data points and draw them onto the graph
        drawn_data_points: tuple[tuple[int, int]] = []
        for x, y in data_points:
            transformed_x = x*width_scaling + width_offset
            transformed_y = 150 - (y*height_scaling + height_offset)
            drawn_data_points.append((transformed_x, transformed_y))
                                    
        canvas.create_line(drawn_data_points, tags="data_points", fill="black")


    def front_end_main_loop(self) -> None:
        """This is the main loop of the front-end of the simulation,
        which will repeatedly execute until the user exits the program."""  
                
        while True:
            # Check for user events and update the pygame window 
            self.check_events()                    
            self.draw_all()
            pygame.display.update()
            
            # Updates the user interface window
            self.set_widgets()
            self.main_window.update()


UserInterface() # Start the program
