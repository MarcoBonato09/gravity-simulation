# %%
from __future__ import annotations
from typing import Literal
import math
import pygame
import tkinter
from tkinter import Canvas, Frame, Button, Label, Entry, Scrollbar, StringVar, Menu
from tkinter.ttk import Combobox, Notebook, Treeview
from tkinter.ttk import Style
import threading
import os

# %%
class Vector:
    """This class represents a 2d vector with x and y components and is used throughout this program."""
    
    # There is minimal input validity checking throughout since this class's use is fully managed by other classes.
    
    def __init__(self, x_component: int|float|None = None, y_component: int|float|None = None,
                magnitude: int|float|None = None, direction: int|float|None = None):
        """This function initializes this vector object, given its components or its magnitude and direction."""
        
        self.__x_component = x_component
        self.__y_component = y_component
        self.__magnitude = magnitude
        self.__direction = direction
        
        # We need either both components or a magnitude and direction to fully describe a vector.
        # Therefore we check if we have either, and then calculate the other components.
        if x_component is not None and y_component is not None:
            self.recalculate_polar_form() # Polar form is the name for the direction and magnitude representation of a vector
        elif magnitude is not None and direction is not None:
            self.recalculate_components()
                        
    
    def recalculate_components(self) -> None:
        """This function recalculates the x and y components of this vector using its magnitude and direction"""
        self.__x_component = self.__magnitude*math.cos(self.__direction)
        self.__y_component = self.__magnitude*math.sin(self.__direction)
        
        
    def recalculate_polar_form(self) -> None:
        """This function recalculates the magnitude and direction of this vector using its x and y components"""
        self.__magnitude = math.sqrt(self.__x_component**2 + self.__y_component**2)
        self.__direction = math.atan2(self.__y_component, self.__x_component)
    
    
    def soften(self, softening_factor: int|float) -> None:
        """This function softens this vector using a given softening factor.
        This function is later used to implementing the softening technique."""
        self.magnitude = math.sqrt(self.magnitude**2 + softening_factor**2)
    
    
    def copy(self) -> Vector:
        """Returns a copy of this vector"""
        return Vector(self.x_component, self.y_component)
    
    
    # These next functions are called magic functions and are here used to define arithmetic operations on vectors
    
    def __add__(self, other_vector: Vector) -> Vector:
        """This function defines the addition operation for a vector.
        When a vector is added to another their components are summed."""
        return Vector(self.x_component+other_vector.x_component, self.y_component+other_vector.y_component)
    
    
    def __sub__(self, other_vector: Vector) -> Vector:
        """This function defines the subtraction operation for a vector.
        When a vector is subtracted from another their components are subtracted from each other."""
        return Vector(self.x_component-other_vector.x_component, self.y_component-other_vector.y_component)
    
    
    def __mul__(self, multiplication_value: int|float|Vector) -> Vector:
        """This function defines the multiplication operation for a vector.
        When a vector is multiplied by a numerical value its magnitude is multiplied by this value.
        If it is multiplied by a vector then the dot product of the two vectors is returned."""
        
        if isinstance(multiplication_value, int|float):
            return Vector(None, None, self.magnitude*multiplication_value, self.direction)
        else:
            other_vector = multiplication_value
            return self.x_component*other_vector.x_component + self.y_component*other_vector.y_component
        
        
    def __rmul__(self, multiplication_value: int|float|Vector) -> Vector:
        """This is the same as __mul__ but for when the multiplication value is to the left of the vector. 
        For example, 5*<vector> calls __rmul__ instead of __mul__. 
        In this case rmul is equivalent to mul, i.e. order of multiplication does not matter."""
        return self.__mul__(multiplication_value)
        
    
    def __truediv__(self, division_factor: int|float) -> Vector:
        """This function defines the division operation for a vector.
        When a vector is divided by a value its magnitude is divded by this value."""
        return Vector(None, None, self.magnitude/division_factor, self.direction)
    
    
    def __floordiv__(self, division_factor: int|float) -> Vector:
        """This function defines the floor division operation for a vector.
        When a vector is floor divided by a value its components are floor divided by this value."""
        return Vector(self.x_component//division_factor, self.y_component//division_factor)
        
    
    def __mod__(self, modulo_factor: int|float) -> Vector:
        """This function defines the modulo operation for a vector.
        A vector % some number sets the components of the vector to the components modulo the number."""
        return Vector(self.x_component%modulo_factor, self.y_component%modulo_factor)    
    
    
    def __neg__(self) -> Vector:
        """This function defines the - symbol in front of a vector (i.e. -vector).
        This returns the vector with its components inverted."""
        return Vector(-self.x_component, -self.y_component)
        
        
    # We use @property to be able to give properties custom setters
    # This is so that vector properties can be re-calculated when other ones are updated
    
    @property
    def components(self) -> tuple[int|float, int|float]:
        """This function returns the x and y components of this vector as a tuple."""
        return (self.x_component, self.y_component)
    
    
    @property
    def x_component(self) -> int|float:
        """Returns the x component of the vector"""
        return self.__x_component
    
    
    @x_component.setter
    def x_component(self, new_x_component: int|float) -> None:
        """This function sets the x component of this vector to the specified value and recalculates its other properties."""
        self.__x_component = new_x_component
        self.recalculate_polar_form()
    
    
    @property
    def y_component(self) -> int|float:
        """Returns the y component of the vector"""
        return self.__y_component
    
    
    @y_component.setter
    def y_component(self, new_y_component: int|float) -> None:
        """This function sets the y component of this vector to the specified value and recalculates its other properties."""
        self.__y_component = new_y_component
        self.recalculate_polar_form()
    
    
    @property
    def magnitude(self) -> int|float:
        """Returns the magnitude of the vector"""
        return self.__magnitude
    
    
    @magnitude.setter
    def magnitude(self, new_magnitude: int|float) -> None:
        """This function sets the magnitude of this vector to the specified value and recalculates its other properties."""
        
        if new_magnitude < 0:
            raise ValueError("A vector's magnitude cannot be less than zero")
        
        self.__magnitude = new_magnitude
        self.recalculate_components()
    
    
    @property
    def direction(self) -> int|float:
        """Returns the direction of the vector in radians"""
        return self.__direction
    
    
    @direction.setter
    def direction(self, new_direction: int|float) -> None:
        """This function sets the direction of this vector to the specified value and recalculates its other properties."""
        self.__direction = new_direction
        self.recalculate_components()
    
        

# %%
class Body:
    """
    Bodies are the basic items that move in our simulation, representing things such as planets or stars.
    They are circular with uniformly distributed masses.
    """
    
    # There is no need for input validity checking throughout as this class's use is completely managed by the Simulation class.
    
    def __init__(self, id: str, mass: int|float, radius: int|float, position: Vector) -> None:
        """
        This function initializes this body's attributes to the given values.
        Other attributes such as an initial velocity are set to a default value and then can be changed by the user.
        All attributes use standard Physics units; mass in kilograms, radius and coordinates in meters, time in seconds, etc. 
        """
                
        self.id = id
        self.mass = mass
        self.radius = radius
        self.previous_position = position
        self.position = position
        self.is_immobile = False
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.engine_acceleration = Vector(0, 0)
        
        
    def step_euler(self, time: int|float) -> None:
        """
        This function uses the Euler method to update the velocity of the body and then move it.
        The higher the time parameter, the further the body will move, but the less accurate its movement.
        If the body is immobile (i.e. the property is_immobile is set to True), 
        then its velocity and acceleration vectors are set to 0.
        """
        
        if not self.is_immobile:
            self.previous_position = self.position.copy()
            self.velocity += (self.acceleration+self.engine_acceleration)*time
            self.position += self.velocity*time
        else:
            self.velocity = Vector(0, 0)
            self.acceleration = Vector(0, 0)
    
        
    @property
    def kinetic_energy(self) -> float:
        """This function returns the body's kinetic energy in joules."""
        return 1/2*self.mass*self.velocity.magnitude**2
    
    
    @property
    def momentum(self) -> Vector:
        """This function returns the momentum of this body as a vector."""
        return self.velocity*self.mass
    
    
    def angular_velocity(self, body_being_rotated_around: Body, time_step: int|float) -> float:
        """This function returns the angular velocity of the body around the body specified in the argument.
        By convention, if it is positive the body is rotating anticlockwise and the opposite is also true."""
        
        # Angular velocity is calculated by finding the angle of the vector pointing from this body to the body being rotated around.
        # This is done for both the body's current and previous position.
        # When these values are subtracted from each other they give a change in angle.
        # This is then divided by the time step to give angular velocity
        previous_angle = (body_being_rotated_around.position-self.previous_position).direction
        current_angle = (body_being_rotated_around.position-self.position).direction
                
        return (current_angle-previous_angle)/time_step
    
    
    def angular_momentum(self, body_being_rotated_around: Body) -> int|float:
        """This function returns the angular momentum of this body around the body passed in.
        This is calculated using the 'cross product' of the body's position and velocity vectors, multiplied by its mass."""
        
        position_vector = self.position - body_being_rotated_around.position # The vector from the body being rotated around to this body
        
        # This formula is based on the definition of angular momentum as the cross product between the velocity and position vector of the body
        return self.mass*(position_vector.x_component*self.velocity.y_component - position_vector.y_component*self.velocity.x_component)
    


# %%
class BarnesHutNode:
    """This class represents one of the nodes in the Barnes-Hut algorithm and the properties of the square in space it represents."""
    
    # There is no need for input validity checking throughout as this class's use is completely managed by the BarnesHutTree class.
    
    def __init__(self, contained_bodies: list[Body], center: Vector, width: int|float) -> None:
        """
        This function initializes this node's properties and calculates other properties (e.g. total mass).
        It also finds the children of this node (assuming the square represented by this node contains more than 1 body)
        """
        
        self.contained_bodies = contained_bodies
        self.num_contained_bodies = len(self.contained_bodies)
        self.center = center
        self.width = width
        self.total_mass = 0
        self.maximum_radius_of_contained_bodies = 0
        self.center_of_mass = Vector(0, 0)
        
        # The values in order at each position are the top left child, top right child, bottom left child, and bottom right child respectively
        self.children = [None, None, None, None]
         
        
class BarnesHutTree:
    """
    This class is responsible for initializing and storing the root node for the Barnes-Hut Tree.
    Operations such as traversal for finding forces or collision detection is handled by the Simulation class.
    """
    
    # There is no need for input validity checking troughout as this class's use is completely managed by the Simulation class
    
    def __init__(self, contained_bodies: list[Body]) -> None:
        """This function initializes the root node of the tree when an object of this class is created."""
        root_center, root_width = self.find_bounding_box(contained_bodies)
        self.root: BarnesHutNode = self.build_node(contained_bodies, root_center, root_width)
        
        
    def find_bounding_box(self, contained_bodies: list[Body]) -> tuple[float, float, int]:
        """
        This function returns the center coordintes and width of the smallest 
        square that contains all the bodies, plus a bit of extra space.
        """
        
        # First we find the minimum and maximum x and y coordinates across all bodies in the simulation
        minimum_coordinates = Vector(min([body.position.x_component for body in contained_bodies]), 
                                     min([body.position.y_component for body in contained_bodies]))
        maximum_coordinates = Vector(max([body.position.x_component for body in contained_bodies]), 
                                     max([body.position.y_component for body in contained_bodies]))
               
        # The center of the square is the average of the maximum and minimum coordinates in each direction
        center = (minimum_coordinates+maximum_coordinates)/2
        
        # The width of the square is the maximum difference between the maximum and minimum coordinates in each direction
        # A little extra width is added (+2) so that bodies right on the edge of the square are not prone to floating point errors
        width = math.ceil(max((maximum_coordinates-minimum_coordinates).components))+2
        
        return center, width
    
    
    def build_node(self, contained_bodies: list[Body], center: Vector, width: int|float) -> BarnesHutNode:
        """This function recursively finds the properties of a node, most importantly its children. This is done recursively.
        This recursion ends at leaf nodes whose squares contain only 0 or 1 bodies."""
        
        node = BarnesHutNode(contained_bodies, center, width)
                        
        if node.num_contained_bodies != 0:
            # These lists will store the bodies in each quadrant of the square represented by this node
            top_left_bodies: list[Body] = []
            top_right_bodies: list[Body] = []
            bottom_left_bodies: list[Body] = []
            bottom_right_bodies: list[Body] = []
            
            for body in contained_bodies:
                # This code block calculates some of the node's important properties
                if body.radius > node.maximum_radius_of_contained_bodies:
                    node.maximum_radius_of_contained_bodies = body.radius
                node.total_mass += body.mass
                node.center_of_mass += body.mass*body.position
                
                # This next block finds the quadrant the body is in and assigns it to one of the quadrant lists 
                
                if body.position.x_component <= node.center.x_component: # If this condition is true, this body is in the left part of the square
                    if body.position.y_component >= node.center.y_component: # If this condition is true, this body is in the top part of the square.
                        top_left_bodies.append(body)
                    else: # Otherwise it is in the bottom part of the square
                        bottom_left_bodies.append(body)
                else: # Otherwise the body is in the right part of the square
                    if body.position.y_component >= node.center.y_component: # If this condition is true, the body is in the top part of the square
                        top_right_bodies.append(body)
                    else: # Otherwise it is in the bottom part of the square
                        bottom_right_bodies.append(body)
            
            node.center_of_mass /= node.total_mass
            
            # We only give this node children if it has more than 1 contained bodies, otherwise we end the recursion
            if node.num_contained_bodies > 1:
                # This next block recursively calls this method to build this node's children
                
                # We use these vectors to find new centers for each of the children (i.e. find the centers of each quadrant)
                top_right_vector = Vector(node.width/4, node.width/4) # The vector pointing from the center of the node to the center of its top right quadrant
                top_left_vector = Vector(-node.width/4, node.width/4) # The vector pointing from the center of the node to the center of its top left quadrant
                
                # Now we find the centers of each quadrant using these vectors.
                top_left_center = node.center + top_left_vector
                top_right_center = node.center + top_right_vector
                bottom_left_center = node.center - top_right_vector # We subtract the top right vector since the bottom left quadrant is directly opposite the top right quadrant
                bottom_right_center = node.center - top_left_vector # We subtract the top left vector since the bottom right quadrant is directly opposite the top left quadrant
                
                node.children[0] = self.build_node(top_left_bodies, top_left_center, node.width/2)
                node.children[1] = self.build_node(top_right_bodies, top_right_center, node.width/2)
                node.children[2] = self.build_node(bottom_left_bodies, bottom_left_center, node.width/2)
                node.children[3] = self.build_node(bottom_right_bodies, bottom_right_center, node.width/2)

        return node
    
    

# %%
class Simulation:
    """This class is responsible for calculating the acceleration on each body, moving each body and handling collisions. 
    If there is a collision between bodies they are merged. 
    The user may choose between the Direct Sum or Barnes-Hut algorithm for both acceleration calculation and collision handling."""
    
    GRAVITATIONAL_CONSTANT = 6.6743e-11
    
    # There is no need for input validity checking troughout as this class's use is completely managed by the UserInterface class
    
    def __init__(self) -> None:
        """This functions initializes the attributes of the simulation to certain default values."""
        
        self.contained_bodies: dict[str, Body] = {} # This is a dictionary that will map the string id of a body to its Body object
        self.time_step = 1 # How long to move bodies along their velocity vectors each frame/simulation step
        self.time_elapsed = 0
        self.theta = 0.5 # The parameter in the Barnes-Hut algorithm. Lower values mean higher accuracy but slower runtime.
        self.epsilon = 0 # The parameter used for the smoothening technique
        # These variables store which algorithms the simulation is currently using for collision detection and acceleration calculation.
        self.acceleration_calculation: Literal["Direct Sum", "Barnes-Hut"] = "Direct Sum"
        self.collision_detection: Literal["Direct Sum (Slow)", "Direct Sum (Optimized)", "Barnes-Hut", "None"] = "Direct Sum (Slow)"

    
    def total_of_property(self, property_name: str, centered_body: Body|None = None) -> int|float:
        """
        This function takes a property name and sums that property over all bodies.
        For example, if property_name is the string "velocity.x_component", then 
        the x component of velocity of all bodies is summed and returned.
        """        
   
        total = 0
        for body in self.contained_bodies.values():
            if property_name == "angular_momentum":
                total += body.angular_momentum(centered_body)
            elif property_name == "angular_velocity":
                total += body.angular_velocity(centered_body, self.time_step)
            else:
                total += eval(f"body.{property_name}")
            
        return total
                
        
    def find_unique_id(self) -> str:
        """This function finds a unique body id not already in use. This can then be assigned to a newly created body.
        This creates ids in the form 'Body n', where n is a number >= 0 and n is the smallest number such that 
        the id 'Body n' does not already exist.
        """
        
        id_number = 0
        # This loop keeps incrementing the id number until the id "body <id_number>" does not exist
        while "Body " + str(id_number) in self.contained_bodies:
            id_number += 1
        return "Body " + str(id_number)
        
        
    def have_bodies_collided(self, body_1: Body, body_2: Body) -> bool:
        """This function returns True or False based on if two bodies have collided."""
                
        # a, b and c are the coefficients of the quadratic equation used to check for inter-frame collisions
        collided_inter_frame = False
        a = (body_1.velocity - body_2.velocity).magnitude**2
        b = 2*((body_1.position-body_2.position)*(body_1.velocity-body_2.velocity))
        c = (body_1.position - body_2.position).magnitude**2-(body_1.radius+body_2.radius)**2
        determinant = b**2-4*a*c
        if determinant >= 0 and a != 0: # These conditions must be met if it was possible for the bodies to collide in-between frames
            # This code block finds the roots of the quadratic equation using the quadratic formula
            root_1 = (-b+math.sqrt(determinant))/(2*a)
            root_2 = (-b-math.sqrt(determinant))/(2*a)
            lesser_root = min(root_1, root_2)
            larger_root = max(root_1, root_2)
            # Check if the time step parameter is between lesser_root and larger_root, which is a necessary condition
            if self.time_step >= lesser_root and self.time_step <= larger_root:
                collided_inter_frame = True
            
        displacement_vector = body_2.position - body_1.position # The vector from body 1 to body 2
        distance = displacement_vector.magnitude # The distance between the centers of the two bodies
            
        # The two bodies have collided if the distance between them is less than the sum of their radii
        # Alternatively they also collide if collided_inter_frame is true
        if distance < body_1.radius+body_2.radius or collided_inter_frame:
            return True
        else:
            return False
        
        
    def merge_bodies(self, body_1: Body, body_2: Body) -> Body:
        """
        This function takes two bodies and returns a merged version of them, which conserves momentum, mass, etc.
        It also automatically removes the two merged bodies from self.contained_bodies and adds the merged one.
        """
        
        merged_body_id = self.find_unique_id()
        merged_body_mass = body_1.mass+body_2.mass
        merged_body_radius = math.sqrt(body_1.radius**2+body_2.radius**2) # This new radius conserves area
                
        # This method of finding the velocity of the merged body conserves momentum in the simulation
        merged_body_velocity = (body_1.velocity*body_1.mass + body_2.velocity*body_2.mass)/(body_1.mass+body_2.mass)
        
        # This way of finding the new coordinates means that the object with the higher mass has more influence over the position.
        merged_body_position = (body_1.position*body_1.mass + body_2.position*body_2.mass)/(body_1.mass+body_2.mass)
            
        # Creates the new body and sets its velocity
        merged_body = Body(merged_body_id, merged_body_mass, merged_body_radius, merged_body_position)
        merged_body.velocity = merged_body_velocity

        # Remove the merged bodies and add the new one to self.contained-bodies        
        self.contained_bodies.pop(body_1.id)
        self.contained_bodies.pop(body_2.id)
        self.contained_bodies[merged_body.id] = merged_body
 
        return merged_body
                
        
    def direct_sum_acceleration_calculation(self) -> None:
        """This function uses the direct sum method to set the acceleration on each body.
        This means finding the force between each pair of bodies and resolving it into acceleration values."""
        
        # Gravity is calculated between pairs of bodies, so all pairs of bodies must be found.
        pairs: list[tuple[Body, Body]] = [] # This list stored tuples of pairs of bodies
        contained_bodies_list = list(self.contained_bodies.values())
        for i in range(len(self.contained_bodies)):
            for j in range(i+1, len(self.contained_bodies)):
                pairs.append((contained_bodies_list[i], contained_bodies_list[j]))
                
                # The acceleration on each body is replaced each time it is calculated
                # Therefore we set it to zero at the start
                contained_bodies_list[i].acceleration = Vector(0, 0)
                contained_bodies_list[j].acceleration = Vector(0, 0)
                
        if len(self.contained_bodies) > 0:
            contained_bodies_list[0].acceleration = Vector(0, 0)

        # This block of code calculates the force between each pair of bodies and adds to each body's acceleration
        for pair in pairs:
            body_1, body_2 = pair[0], pair[1]
            
            # We now find the acceleration vector due to the gravitational attraction between this pair of bodies on each body
            
            # Find the softened displacement vectors and distance between the centers of the two bodies 
            displacement_vector_1 = body_2.position - body_1.position # The vector from body 1 to body 2
            displacement_vector_1.soften(self.epsilon)
            displacement_vector_2 = -displacement_vector_1 # The softened vector from body 2 to body 1
            softened_distance = displacement_vector_1.magnitude
                        
            # The magnitude of the acceleration due to gravity on each body using Newton's second law (a = F/m) and Newton's law of gravitation
            # Sotening is also used
            acceleration_magnitude_1 = self.GRAVITATIONAL_CONSTANT*body_2.mass/softened_distance**2
            acceleration_magnitude_2 = self.GRAVITATIONAL_CONSTANT*body_1.mass/softened_distance**2
            
            # Since the acceleration due to gravity points from one body towards the other, its direction is the same as the displacement vector
            acceleration_1 = Vector(None, None, acceleration_magnitude_1, displacement_vector_1.direction)
            acceleration_2 = Vector(None, None, acceleration_magnitude_2, displacement_vector_2.direction)
    
            # Add these acceleration vectors to each body in their respective directions
            body_1.acceleration += acceleration_1
            body_2.acceleration += acceleration_2
        
          
    def direct_sum_collision_handling(self) -> None:
        """This function detects and handles collisions (by merging) between bodies. 
        It does this by repeatedly checking for collisions between every pair of bodies 
        until there are no more collisions. Collided bodies merge while conserving momentum, mass, etc."""
                    
        merge_occured = True
        while merge_occured: # Pairs of bodies are iterated through repeatedly until no bodies were merged
            contained_bodies_list = list(self.contained_bodies.values())
            removed_bodies = set() # Keeps track of removed bodies: we skip over them when they are encountered
            
            merge_occured = False
            # Iterate over every pair of bodies and check for a collision
            for i in range(len(contained_bodies_list)):
                body_1 = contained_bodies_list[i]
                if body_1 in removed_bodies: continue # Do not consider a body which has been removed
                
                for j in range(i+1, len(contained_bodies_list)):
                    body_2 = contained_bodies_list[j]
                    if body_2 in removed_bodies: continue # Do not consider a body which has been removed
                    elif self.have_bodies_collided(body_1, body_2):
                        merge_occured = True
                        removed_bodies.add(body_1)
                        removed_bodies.add(body_2)  
                        body_1 = self.merge_bodies(body_1, body_2) # Body 1 is replaced with the merged body
                        
                        
    def optimized_direct_sum_collision_handling(self):
        """
        This function is similar to direct_sum_collision_handling(), but is optimized. 
        It still has the same time complexity of O(n²). It works by merging all the bodies
        that collided with a body, and then immediately adding the final merged body to a queue.
        This means that the merged body can immediately be checked for more collisions, unlike in the 
        other function where the user has to wait for the outer while loop to start again.
        """
        
        queue = list(self.contained_bodies.values())
        
        while queue: # Keep going until the queue is empty (meaning nothing collided)
            body_1 = queue.pop()
            index = 0
            merge_occured = False
            while index < len(queue): # Iterate over the elements of the queue
                body_2 = queue[index]
                
                if self.have_bodies_collided(body_1, body_2):
                    merge_occured = True
                    body_1 = self.merge_bodies(body_1, body_2) # Replace body 1 with the merged body
                    
                    # Remove the queue at this index since body_2 was "absorbed" by body 1.
                    # This also means changing the index, since body_2 is no longer in the queue.
                    del queue[index]
                    index -= 1
                
                index += 1 # Increment the index pointer to keep iterating throutgh the queue
                
            if merge_occured: # Only add the merged body back to the queue if it has merged with something
                queue.append(body_1)
                

    def barnes_hut_acceleration_calculation(self) -> None:
        """This function uses the Barnes-Hut algorithm to set the acceleration on each body.
        This means approximating the acceleration on a body by clustering distant groups of bodies."""
        
        self.merge_bodies_in_same_location()
        tree = BarnesHutTree(list(self.contained_bodies.values()))
        
        for body in self.contained_bodies.values():
            # A body's acceleration is replaced each time it is calculated, so at first we set it to zero
            body.acceleration = Vector(0, 0)
            # The quadtree is now traversed in a depth-first search. 
            stack = [tree.root] # The traversal uses a stack (meaning a depth-first search) and starts with the root of the tree
            while stack:
                front_node = stack.pop()
                                
                # Checks that the front node has a non-zero number of bodies
                # Also checks that it doesn't contain only the body whose acceleration is being calculated
                if front_node.num_contained_bodies != 0 and not (front_node.num_contained_bodies == 1 and front_node.contained_bodies[0] == body):
                    # Find the softened displacement vector between the body and the node square's center of mass
                    displacement_vector = front_node.center_of_mass - body.position # The vector from the body to the node's center of mass
                    displacement_vector.soften(self.epsilon)
                    softened_distance = displacement_vector.magnitude
                    
                    # If the node square's width divided by the distance is less than theta, this node is considered far away
                    # Therefore its effect on the body will be approximated using the node's center of mass and total mass
                    # Otherwise, if the node contains only 1 body, we simply calculate the acceleration on body using the body contained in the node.
                    if (softened_distance != 0 and front_node.width/softened_distance < self.theta) or front_node.num_contained_bodies == 1:
                        # The approximated magnitude of the acceleration due to gravity on the body due to the bodies in the node's square
                        acceleration_magnitude = (self.GRAVITATIONAL_CONSTANT*front_node.total_mass)/softened_distance**2
                        
                        # Remember that the direction of the acceleration vector is the same as the displacement vector
                        # This is because the body is attracted towards the node's center of mass
                        acceleration_on_body = Vector(None, None, acceleration_magnitude, displacement_vector.direction)
                        body.acceleration += acceleration_on_body # Add this acceleration vector to the body
                        
                    elif front_node.num_contained_bodies != 1: # This node is not considered far away. Its children will be enqueued.
                        for child in front_node.children:
                            stack.append(child)
        

    def barnes_hut_collision_handling(self) -> None:
        """This function detects and handles collisions (by merging) between bodies. 
        It does this by checking for collisions between bodies and excluding bodies which are too far away from another to collide with it. 
        Merging bodies is repeated until there are no more collisions. Collided bodies merge while conserving momentum, mass, etc."""
        
        merge_occured = True
        while merge_occured: # We keep repeating the process until no more merges occur
            merge_occured = False
            removed_bodies = set() # Keeps track of which bodies have been removed from self.contained_bodies
            self.merge_bodies_in_same_location() # Bodies have change locations due to merging, so two could be in the same place
            contained_bodies_list = list(self.contained_bodies.values())
            tree = BarnesHutTree(contained_bodies_list)

            for body_1 in contained_bodies_list:
                if body_1 in removed_bodies: continue # Do not consider a body that has been removed
                
                stack = [tree.root] # Start the depth first search with the root node
                while stack:
                    front_node = stack.pop()
                    # Checks if the currently searched node has only one body (which is not body_1 and has not been removed)
                    # If so, the collision between body_1 and that body is checked for as usual
                    if front_node.num_contained_bodies == 1 and front_node.contained_bodies[0] != body_1 and front_node.contained_bodies[0] not in removed_bodies:
                        body_2 = front_node.contained_bodies[0]
                        if self.have_bodies_collided(body_1, body_2):
                            merge_occured = True
                            removed_bodies.add(body_1)
                            removed_bodies.add(body_2)
                            body_1 = self.merge_bodies(body_1, body_2) # body_1 is replaced with the merged body
                            
                    # Otherwise, check if it's possible for body_1 to collide with the bodies contained in front_node's subtree
                    elif front_node.num_contained_bodies > 1:
                        # Find the minimum distance from a point in the node's square to the body
                        displacement_vector = body_1.position - front_node.center # The vector from the center of the node to the body
                        displacement_vector.x_component = max(0, abs(displacement_vector.x_component) - front_node.width/2)
                        displacement_vector.y_component = max(0, abs(displacement_vector.y_component) - front_node.width/2)
                        minimum_distance = displacement_vector.magnitude   

                        # It's only possible for body_1 to collide with a body contained in front_node if this condition is met
                        # If this is true then we enqueue the node's children to the stack, otherwise not
                        if minimum_distance <= body_1.radius+front_node.maximum_radius_of_contained_bodies:
                            for child in front_node.children:
                                stack.append(child)
            
        
    def merge_bodies_in_same_location(self) -> None:
        """This function does as it says, and merges bodies with the same coordinates, while conserving mass, momentum etc.
        This is needed as the Barnes-Hut clustering algorithm will recurse infinitely if two bodies are in the same location."""
        
        location_dictionary: dict[Vector] = {} # Maps a position vector to a body, can be used to quickly see if two bodies are in the same location
    
        # If there are two bodies the merged body will be mapped to the same set of coordinates
        for body_1 in self.contained_bodies.values():
            if body_1.position in location_dictionary: # If so, there is already a body at those coordinates 
                body_2 = location_dictionary[body_1.position] # Keys are a tuple of (x, y) coordinates
                merged_body = self.merge_bodies(body_1, body_2)
                location_dictionary[body_1.position] = merged_body
            else:
                location_dictionary[body_1.position] = body_1
            
            
    def step_euler(self) -> None:
        """This function steps each body in the simulation using the Euler method."""
        
        self.time_elapsed += self.time_step
        for body in self.contained_bodies.values():
            body.step_euler(self.time_step)


    def find_center_of_mass(self) -> Body:
        """This function returns the center of mass of the simulation as a body. 
        This is used to set the centered object of the simulation to this body, 
        when the center of mass is selected as the centered body for the simulation."""
        
        center_of_mass = Vector(0, 0)
        total_mass = 0
        
        for body in self.contained_bodies.values():
            center_of_mass += body.position*body.mass
            total_mass += body.mass
            
        center_of_mass /= total_mass
                        
        return Body("Center of mass", 0, 0, center_of_mass)
            

    def step(self) -> None:
        """This function combines acceleration calculation, collision detection and the Euler method
        to move bodies in the simulation."""
        
        match self.collision_detection: # Executes the collision detection algorithm chosen by the user
            case "Barnes-Hut": self.barnes_hut_collision_handling()
            case "Direct Sum (Slow)": self.direct_sum_collision_handling()
            case "Direct Sum (Optimized)": self.optimized_direct_sum_collision_handling()
            case "None": pass
            
        match self.acceleration_calculation: # Executes the acceleration calculation algorithm chosen by the user
            case "Barnes-Hut": self.barnes_hut_acceleration_calculation()
            case "Direct Sum": self.direct_sum_acceleration_calculation()
 
        self.step_euler() # Moves all the bodies in the simulation according to the Euler method


    def add_body(self, mass: int|float, radius: int|float, position: Vector) -> Body:
        """This function is used to add a body to the simulation. It is used to automatically find an id for a new body 
        and to update self.contained_bodies with the new information. It is also returned for use in the PygameWindow class."""
        
        new_body = Body(self.find_unique_id(), mass, radius, position)
        self.contained_bodies[new_body.id] = new_body
        return new_body



# %%
class PygameWindow(Simulation):
    """This class handles the use of the Pygame library in the project. It is responsible for drawing the grid, the bodies on it, etc. 
    It is also responsible for checking user events (e.g. zooming in or out of the grid or clicking on a body to select it)."""
    
    
    def __init__(self) -> None:
        """This function initializes this object and all of its parameters. 
        It also initializes the pygame window, making it appear to the user."""
        
        super().__init__() # Initializes the parent simulation class
        
        pygame.init() # Initializes the pygame library
        screen_information = pygame.display.Info()
        self.window_dimensions = Vector(screen_information.current_w, screen_information.current_h)
        
        # Initialize the pygame window to be fullscreen and scaled to the user's monitor size
        self.pygame_window = pygame.display.set_mode(self.window_dimensions.components, pygame.FULLSCREEN|pygame.SCALED)
        
        # Initializes all the variables or parameters in this class.
        self.clock = pygame.time.Clock()
        self.grid_square_length = 30 # The length in pixels of the small grid squares drawn
        self.arrow_length_factor = 1 # The length of the arrows drawn on bodies are multiplied by this value
        self.zoom_factor = 1
        self.offset = Vector(0, 0) # This offset is based on the user dragging the grid
        self.mouse_held = False
        self.adding_body_mode = False # If this is enabled, the user can click on the screen to add a body
        self.centered_body = Body("Origin", 0, 0, Vector(0, 0))
        self.selected_body = None
        self.show_trail = False # A trail is only shown if this value is true
        self.trail_points: list[tuple[Vector, Vector]] = [] # Stores the previous positions of the selected body to draw a trail
        self.body_to_orbit_around: Body|None = None
        self.semi_major_axis_length = 0
        self.paused = True
        self.frames_per_second = 60
    
        
    def check_events(self) -> None:
        """This function checks for events like the user clicking on the screen or dragging their mouse, and 
        resolves them into actions. For example, if a user clicks on a body then that body is selected."""
        
        for event in pygame.event.get(): # Each event is a pygame event object with certain attributes
            if event.type == pygame.MOUSEWHEEL: # The user has used their mousewheel, so we will change the zoom of the grid
                # event.y is positive or negative based on if the user has moved the wheel up or down.
                # Using this, the formula 1+event.y/10 gives a percentage scale factor to scale our grid by.
                zoom_percentage_multiplier = 1+event.y/10
                self.zoom_factor *= zoom_percentage_multiplier
                self.offset *= zoom_percentage_multiplier # x and y offsets also must be scaled so that the user is still looking at the same spot on the grid
                
                # We also apply this scaling factor to the grid square length.
                # If the grid squares become too small or too large, we reset them to the other end of the size specturm.
                self.grid_square_length = int(self.grid_square_length*zoom_percentage_multiplier)
                if self.grid_square_length < 10: self.grid_square_length = 30
                elif self.grid_square_length > 30: self.grid_square_length = 10
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # The user has clicked their left mouse button
                self.mouse_held = True
                
                # We convert the user's click position on the grid to simulation coordinates, which we can use to see if they clicked on a body
                mouse_position = Vector(*pygame.mouse.get_pos())
                simulation_coordinates = (mouse_position-self.offset-self.window_dimensions/2)
                simulation_coordinates.y_component *= -1 # This is needed because pygame has a flipped y axis compared to our simulation
                simulation_coordinates = simulation_coordinates/self.zoom_factor+self.centered_body.position
                
                if self.adding_body_mode: # The user was in adding body mode and they have clicked on the screen, so we add a body where they clicked.
                    new_body_mass = 10**14/self.zoom_factor**2 # Note the squared; this means the mass of the new body grows proportionally its displayed area
                    new_body_radius = 20/self.zoom_factor # Using this, the user will add a body of constant size relative to their screen no matter the zoom factor
                    self.selected_body = self.add_body(new_body_mass, new_body_radius, simulation_coordinates)
                    self.adding_body_mode = False
                else:
                    for body in self.contained_bodies.values(): # Check if the user has clicked on a body which is not the selected body
                        displacement_vector = simulation_coordinates - body.position
                        distance = displacement_vector.magnitude # The distance between the click location and the current body being iterated on
                        
                        if distance <= body.radius and body != self.selected_body: # Checks if the user has clicked on this body
                            # If so, we change the selected body to what they selected and clear the trail points.
                            self.selected_body = body
                            self.trail_points = []
                            break
            elif event.type == pygame.MOUSEBUTTONUP: # The user has let go of their left mouse button
                self.mouse_held = False
            elif event.type == pygame.MOUSEMOTION and self.mouse_held: # The user is dragging the screen, so we need to drag the grid
                self.offset += Vector(*event.rel) # event.rel is a tuple describing how much the user dragged their mouse by
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: # Un/pause the simulation when the user presses their space bar
                self.paused = not self.paused


    def draw_pygame_window(self) -> None:
        """This function draws everything onto the pygame screen; the bodies, grid, and any other miscellaneous items like a trail or an orbit preview.
        It clears the pygame window and then calls draw_grid(), draw_bodies() and draw_extra() to add everything to the screen"""
        
        # First we fill the screen with black, as everything is redrawn onto the screen many times a second.
        self.pygame_window.fill("black")
                
        # Next, we draw everything onto the screen. 
        # Note the order: if we were to call draw_grid() after draw_bodies() then grid lines would be displayed on top of bodies
        self.draw_grid()
        self.draw_bodies()
        self.draw_extra()


    def draw_grid(self) -> None:
        """This function draws the grid onto the pygame screen. It also draws the grid indicator onto the screen, 
        showing the size of a large grid square in various space units."""
                
        # Draw the small and large grid squares. A large grid square contains a 5 by 5 square of small grid squares.
        self.draw_grid_lines(self.grid_square_length, "#141414")
        self.draw_grid_lines(self.grid_square_length*5, "#444444")
            
        # Draw the center grid lines representing the origin, which are the brightest lines.
        pygame.draw.line(self.pygame_window, "#BBBBBB",
                        (self.window_dimensions.x_component//2+self.offset.x_component, 0), 
                        (self.window_dimensions.x_component//2+self.offset.x_component, self.window_dimensions.y_component))
        pygame.draw.line(self.pygame_window, "#BBBBBB", 
                        (0, self.window_dimensions.y_component//2+self.offset.y_component), 
                        (self.window_dimensions.x_component, self.window_dimensions.y_component//2+self.offset.y_component))
        
        # Draw the space indicator showing the size of a large grid square in various space units
        self.draw_space_indicator(self.grid_square_length*5)


    def draw_grid_lines(self, square_length: int, color: str) -> None:
        """This function draws a set of horizontal and vertical lines forming a grid using the given parameters."""
        
        # First we find the x coordinate of the first vertical line and the first y coordinate of the first horizontal line
        grid_start_point = (self.window_dimensions//2+self.offset)%square_length
        grid_x_start_point, grid_y_start_point = map(int, grid_start_point.components)

        # Next we use two for loops and iterate over the coordinates of each line. We then draw the line using pygame.draw.line()
        for x_coordinate in range(grid_x_start_point, self.window_dimensions.x_component, square_length): # Draw the vertical lines at different x coordinates
            start_coordinate = (x_coordinate, 0)
            end_coordinate = (x_coordinate, self.window_dimensions.y_component)
            pygame.draw.line(self.pygame_window, color, start_coordinate, end_coordinate)
        
        for y_coordinate in range(grid_y_start_point, self.window_dimensions.y_component, square_length): # Draw the horizontal lines at different y coordinates
            start_coordinate = (0, y_coordinate)
            end_coordinate = (self.window_dimensions.x_component, y_coordinate)
            pygame.draw.line(self.pygame_window, color, start_coordinate, end_coordinate)


    def draw_space_indicator(self, square_length: int) -> None:
        """This function finds the bottom-right most grid square and draws a bar around it showing, in text, 
        its length in different units (meters, kilometers, astronomical units and light seconds)."""
        
        # Find the (x, y) coordinates of the bottom right point of the bottom-right most grid square
        grid_start_point = (self.window_dimensions//2+self.offset)%square_length
        bottom_right = self.window_dimensions-(self.window_dimensions-grid_start_point)%square_length
        x, y = bottom_right.components
        
        pygame.draw.lines(self.pygame_window, "white", False, ((x, y), (x, y-square_length//4), (x-square_length, y-square_length//4), (x-square_length, y)))
        
        font_used = pygame.font.Font(size=30)
        square_length /= self.zoom_factor # We now convert square length to its simulation size in meters
        maximum_text_width, maximum_text_height = font_used.size(f"{square_length/149597870700:.3e} AU")
        
        texts_to_display = [font_used.render(f"{square_length:.3e} m", 0, "grey"), 
                            font_used.render(f"{square_length/1000:.3e} km", 0, "grey"), 
                            font_used.render(f"{square_length/149597870700:.3e} AU", 0, "grey"), 
                           font_used.render(f"{square_length/299792458:.3e} Ls", 0, "grey"), ]
        
        for index, text in enumerate(texts_to_display):
            self.pygame_window.blit(text, (x-maximum_text_width, y-square_length//4-maximum_text_height*(index+1)))
        
                
    def draw_bodies(self) -> None:
        """This function draws all the bodies in the simulation onto the screen.
        It also draws any extra parts of bodies like vector arrows drawn onto them,
        orbit previews, trail points or a preview of a body being added to the simulation.""" 
        
        # Draw the bodies in the simulation onto the screen
        for body in self.contained_bodies.values():
            # Convert the coordinates and radius of the body to the pygame coordinates and pygame drawn radius
            # Again, the multiplication by -1 is because pygame and the simulation have flipped y axes
            pygame_coordinates = body.position*self.zoom_factor-self.centered_body.position*self.zoom_factor
            pygame_coordinates.y_component *= -1
            pygame_coordinates += self.window_dimensions//2+self.offset
            drawn_radius = body.radius*self.zoom_factor
            
            # By default the color is blue, but if this body is selected it is drawn in yellow instead
            color = "Blue"
            if body == self.selected_body:
                color = "yellow"
                
            pygame.draw.circle(self.pygame_window, color, pygame_coordinates.components, drawn_radius)
        
        # We now add any extra items to the selected body, if there is a body selected.
        # This includes its trail, a preview of its orbit, and arrows representing vectors acting on it.
        if self.selected_body:
            # If the user has enabled trails on the selected body, we draw its trail
            if self.show_trail:
                self.draw_trail_points()
            
            # If the user has inputted values for a stable orbit, we will draw a preview for the selected body's orbit
            if self.semi_major_axis_length and self.body_to_orbit_around:
                self.draw_orbit_preview()
            
            # Draw vector arrows onto the selected body, showing its velocity, acceleration and engine acceleration.
            if not self.selected_body.is_immobile:
                # .copy() is needed here or draw_arrow_on_selected_body() would modify the vectors we pass in! This would be a logic error
                self.draw_arrow_on_selected_body("red", self.selected_body.velocity.copy())
                self.draw_arrow_on_selected_body("purple", self.selected_body.acceleration.copy())
                self.draw_arrow_on_selected_body("green", self.selected_body.engine_acceleration.copy())
            
        # Finally, the user will be shown a preview of the body they are adding if they are adding one
        if self.adding_body_mode:
            pygame.draw.circle(self.pygame_window, "blue", pygame.mouse.get_pos(), 20)
            
            
    def draw_arrow_on_selected_body(self, color: str, vector: Vector) -> None:
        """This function draws an arrow representing a vector on the selected body, when given the vector's components."""
        
        if vector.magnitude == 0: # The arrow will have no length so we can skip over drawing it
            return
        else:
            vector.y_component *= -1 # Remember that pygame has a flipped y axis so we need to invert the y component
        
        # We convert the selected body's coordinates (which will be the base of our arrow) to pygame coordinates
        pygame_coordinates = (self.selected_body.position - self.centered_body.position)*self.zoom_factor
        pygame_coordinates.y_component *= -1
        pygame_coordinates += self.window_dimensions//2 + self.offset
        
        # We calculate the coordinates of the arrow tip. Note how the components are scaled by both the arrow scale facctor and the zoom factor.
        arrow_tip_pygame_coordinates = pygame_coordinates+vector*self.arrow_length_factor*self.zoom_factor

        pygame.draw.line(self.pygame_window, color, # Draw the stem of the arrow (i.e. the part without the arrowhead)
                         pygame_coordinates.components, 
                         arrow_tip_pygame_coordinates.components,
                         math.ceil(0.5*self.selected_body.radius*self.zoom_factor))
        
        # Now we draw the arrowhead. This arrowhead does not scale with component size, so first we scale down the components to a set size.
        length = self.selected_body.radius*self.zoom_factor # The components will be scaled so that the magnitude of the vector has this value
        vector.y_component *= -1
        normalized_vector = vector/vector.magnitude*length
        
        pygame.draw.polygon(self.pygame_window, 
                            color,
                            ((arrow_tip_pygame_coordinates.x_component-normalized_vector.y_component, arrow_tip_pygame_coordinates.y_component-normalized_vector.x_component),
                            (arrow_tip_pygame_coordinates.x_component+normalized_vector.y_component, arrow_tip_pygame_coordinates.y_component+normalized_vector.x_component),
                            (arrow_tip_pygame_coordinates.x_component+normalized_vector.x_component, arrow_tip_pygame_coordinates.y_component-normalized_vector.y_component)))


    def draw_orbit_preview(self) -> None:
        """This function draws the orbit preview for the selected body IF a user has inputted both a semi major axis length and a body to orbit around. 
        Note that this preview depends on the assumptions made in the Vis-Viva equation for calculating a stable orbit, 
        so if any of these are broken this preview may not be accurate."""
                    
        # Calculate the minor axis length, i.e. the width of the ellipse.
        displacement_vector = self.body_to_orbit_around.position - self.selected_body.position
        apoapsis = displacement_vector.magnitude
        periapsis = 2*self.semi_major_axis_length-apoapsis
        
        try:
            minor_axis_length = 2*math.sqrt(apoapsis*periapsis)
        except ValueError: # The length of the semi major axis is impossibly short, we will not display a preview
            return
        
        # We now draw and rotate the ellipse.
        theta = displacement_vector.direction # The angle to rotate the ellipse by
        
        # Now using maths we find the position vector of the center of the ellipse
        ellipse_center = self.zoom_factor*(self.selected_body.position-self.centered_body.position+Vector(None, None, self.semi_major_axis_length, theta))
        ellipse_center.y_component *= -1
        
        if self.semi_major_axis_length < apoapsis*5: # If the input is too large pygame will crash so we need to check for this
            # Now we create, rotate nd draw the ellipse onto the screen
            ellipse_surface = pygame.Surface((2*self.semi_major_axis_length*self.zoom_factor, minor_axis_length*self.zoom_factor), pygame.SRCALPHA) # This is what the ellipse will be drawn onto
            pygame.draw.ellipse(ellipse_surface, "green", (0, 0, 2*self.semi_major_axis_length*self.zoom_factor, minor_axis_length*self.zoom_factor), 5)
            ellipse_surface = pygame.transform.rotate(ellipse_surface, math.degrees(theta))
            ellipse_dimensions = Vector(*ellipse_surface.get_rect().size)
            self.pygame_window.blit(ellipse_surface, (ellipse_center-ellipse_dimensions/2+self.offset+self.window_dimensions//2).components) # Add the ellipe onto the screen, accounting for its dimensions and offsets
                
    
    def draw_trail_points(self) -> None:
        if len(self.trail_points) >= 2: # There must be at least two positions to draw a line between them
            drawn_points: list[tuple[int, int]] = [] # Remember trail points are a list of tuples of (x_coordinate, y_coordinate, center_x_offset, center_y_offset)
            
            # We now transform the trail points (which are in simulation coordinates) to pygame coordinates.
            for position, centering_offset in self.trail_points:
                pygame_coordinates = position-centering_offset
                pygame_coordinates.y_component *= -1
                pygame_coordinates = pygame_coordinates*self.zoom_factor+self.window_dimensions//2+self.offset
                
                drawn_points.append(pygame_coordinates.components)
                
            pygame.draw.lines(self.pygame_window, "white", False, drawn_points) # Draws the trail where the selected body has been
                
                
    def draw_extra(self) -> None:
        """This function draws miscelanneous items onto the pygame window, namely the paused icon and the fps counter."""
        
        # If the simulation is paused we will display a paused icon on the screen so the user knows the simulation is paused
        if self.paused:
            pygame.draw.rect(self.pygame_window, "grey", (50, 50, 40, 150))
            pygame.draw.rect(self.pygame_window, "grey", (125, 50, 40, 150))

        # Next, we draw an indicator showing the current fps (frames per second)
        # This is already viewable and editable in the user interface but may not be accurate if turned up too high
        font = pygame.font.Font(size=30)
        text = font.render(f"{int(self.clock.get_fps())} fps", 0, "grey")
        self.pygame_window.blit(text, (self.window_dimensions.x_component-100, 50))
                        
                
    def pygame_main_loop(self) -> None:
        """This is the main loop of the pygame window, which runs separate from the tkinter window. 
        We step bodies and also perform a few checks that cannot be done in check_events() 
        (as they should be done only after each step but check_events() runs repeatedly)."""
        
        while True:
            if not self.paused:
                self.step()  
                
                # Adds the selected body's coordinates to the trail points if trails are enabled
                if not self.show_trail or self.selected_body is None:
                    self.trail_points = []
                else:
                    self.trail_points.append([self.selected_body.position, self.centered_body.position])     
                    
                # Recalculates the center of mass of the simulation if the user has chosen this as their centered object
                if self.centered_body.id == "Center of mass":
                    self.centered_body = self.find_center_of_mass()
                  
            self.clock.tick(self.frames_per_second) # Wait a bit of time to ensure a constant frame rate at self.frames_per_second
            
            
    def remove_body(self) -> None:
        """A function that removes the selected body from the simulation (the only way of removing bodies)."""
        
        self.contained_bodies.pop(self.selected_body.id)
        if self.centered_body == self.selected_body: 
            self.centered_body = None
        self.selected_body = None


    def generate_stable_orbit(self) -> None:
        """This function generates a stable orbit velocity for the selected body around the body to orbit around at the 
        semi major axis length specified. This uses the Vis Viva equation and the assumptions that come with it."""
        
        # If any of these conditions are met this subprogram is exited
        if self.semi_major_axis_length is None or self.body_to_orbit_around is None or self.selected_body is None:
            return
        
        # Now we calculate the stable velocity vector and give it to the selected body
        displacement_vector = self.body_to_orbit_around.position - self.selected_body.position # The vector pointing from the orbiting body to the body to orbit around
        distance = displacement_vector.magnitude # The distance between the centers of the two bodies using Pythagoras
        
        # If this condition is true, the inputted semi major axis length is impossibly small, so we exit the subprogram
        if self.semi_major_axis_length < distance/2:
            return
    
        # We now find the magnitude and direction of the body's stable orbit velocity vector
        velocity_magnitude = math.sqrt(self.GRAVITATIONAL_CONSTANT*self.body_to_orbit_around.mass*(2/distance-1/self.semi_major_axis_length))
        velocity_direction = math.pi/2 + displacement_vector.direction
        
        self.selected_body.velocity = Vector(None, None, velocity_magnitude, velocity_direction)
        
        # We now clear the user's input to the orbit generator menu
        self.body_to_orbit_around = None
        self.semi_major_axis_length = 0



# %%
class TkinterWindow(PygameWindow):
    """This class manages the tkinter window which displays the user interface. Here the user can modify various simulation parameters, 
    see value graphs and also read a manual explaining how to use the program. """
    
    # A set containing attributes of bodies or total attributes that have include space units
    # A set (equivalent to a hash map) is used to quickly check if a string is in the set
    ATTRIBUTES_WITH_SPACE = set((
        "semi_major_axis_length", "position.x_component", "position.y_component", "radius", "velocity.magnitude", 
        "velocity.x_component", "velocity.y_component", "acceleration.magnitude", "acceleration.x_component", 
        "acceleration.y_component", "engine_acceleration.magnitude", "engine_acceleration.x_component", 
        "engine_acceleration.y_component", "momentum.magnitude", "momentum.x_component", "momentum.y_component", 
        "kinetic_energy", "angular_momentum", "epsilon"))
    
    
    def __init__(self) -> None:
        super().__init__() # Initializes the parent PygameWindow class
        
        # self.master is the main "parent" of the tkinter window
        self.master = tkinter.Tk()
        self.master.attributes("-topmost", True) # This specifies that the window should appear on top of all other windows
        self.master.protocol("WM_DELETE_WINDOW", lambda: os.kill(os.getpid(), 9)) # Sets the close window button to execute pygame.quit() when clicked
        self.master.resizable(False, False) # Sadly tkinter has a strange bug that triggers when resizing the window which crashes our program, so we disable resizing

        self.space_units = "Meters"
        self.style = Style()
        
        # StringVars are special objects assigned to textboxes which can be set to change the value shown in the textbox.
        # They are also used to retrieve the value inputted to the textbox.
        self.stringvars = {
            "time_elapsed": StringVar(self.master, f"{self.time_elapsed:.3e}"),
            "theta": StringVar(self.master, f"{self.theta:.3e}"), 
            "epsilon": StringVar(self.master, f"{self.epsilon:.3e}"), 
            "time_step": StringVar(self.master, f"{self.time_step:.3e}"), 
            "arrow_scale_factor": StringVar(self.master, f"{self.arrow_length_factor:.3e}"), 
            "frames_per_second": StringVar(self.master, f"{self.frames_per_second:.3e}"), 
            "id": StringVar(self.master, "-"), 
            "position.x_component": StringVar(self.master, "-"), 
            "position.y_component": StringVar(self.master, "-"),
            "mass": StringVar(self.master, "-"),
            "radius": StringVar(self.master, "-"),
            "is_immobile": StringVar(self.master, "-"),
            "velocity.direction": StringVar(self.master, "-"),
            "velocity.magnitude": StringVar(self.master, "-"),
            "velocity.x_component": StringVar(self.master, "-"),
            "velocity.y_component": StringVar(self.master, "-"),
            "acceleration.direction": StringVar(self.master, "-"),
            "acceleration.magnitude": StringVar(self.master, "-"),
            "acceleration.x_component": StringVar(self.master, "-"),
            "acceleration.y_component": StringVar(self.master, "-"),
            "engine_acceleration.direction": StringVar(self.master, "-"),
            "engine_acceleration.magnitude": StringVar(self.master, "-"),
            "engine_acceleration.x_component": StringVar(self.master, "-"),
            "engine_acceleration.y_component": StringVar(self.master, "-"),
            "momentum.magnitude": StringVar(self.master, "-"),
            "momentum.x_component": StringVar(self.master, "-"),
            "momentum.y_component": StringVar(self.master, "-"),
            "kinetic_energy": StringVar(self.master, "-"),
            "angular_velocity": StringVar(self.master, "-"),
            "angular_momentum": StringVar(self.master, "-"), 
            "semi_major_axis_length": StringVar(self.master, "0")}
        
        self.dropdowns = {} # This maps names of dropdown menus to their tkinter Combobox() widget
        
        # This list is constructed later, and contains tuples of information about each graph.
        # Each tuple contains, in order:
        #   - The canvas the graph is drawn on, 
        #   - The stringvar that stores the current value of the parameter displayed
        #   - The stringvar that stores the displayed value for the time range of the graph
        #   - The time range of the graph (how far back in time data points are shown)
        #   - A list of tuples of data points, in the form (simulation time, parameter value)
        #   - The attribute name being graphed
        self.graphs: list[list[Canvas, StringVar, StringVar, int, list[tuple[Vector, Vector]], str]] = []
                        
        # This next part constructs the user interface. 
        # A notebook is the widget at the top, giving multiple windows to choose from
        main_notebook = Notebook(self.master)
        
        selected_body_notebook = Notebook(main_notebook)
        selected_body_notebook.add(child=self.selected_body_info_window(main_notebook), text="Attributes")
        selected_body_notebook.add(child=self.selected_body_orbit_generator_window(main_notebook), text="Stable orbit generator")
        
        for name, widget in (("Basic buttons", self.buttons_window(main_notebook)), 
                             ("Simulation parameters", self.parameters_window(main_notebook)), 
                             ("Selected body menu", selected_body_notebook),
                             ("Graphs", self.graphs_window(main_notebook)),
                             ("Manual", self.manual_window(main_notebook))):
            main_notebook.add(child=widget, text=name)
        
        main_notebook.pack() # .pack() is one way of adding widgets to the screen
        
        # We now start the main loop of the program
        self.main_loop() 


    def scrollable_frame(self, container, **kwargs) -> tuple[Scrollbar, Canvas, Frame]:
        """This function returns a scrollbar, canvas and frame that are combined to allow for a scrollable widget in tkinter."""
        
        # This code block was taken from the internet.
        container_background = Canvas(container, **kwargs)
        scrollbar = Scrollbar(container, orient="vertical", command=container_background.yview)
        container = Frame(container_background)
        container.bind("<Configure>", lambda event: container_background.configure(scrollregion=container_background.bbox("all")))
        container_background.create_window((0, 0), window=container, anchor="nw")
        container_background.configure(yscrollcommand=scrollbar.set)
        
        return scrollbar, container_background, container


    def label_to_attribute_name(self, label_text: str) -> str:
        """
        This function converts the label text for a parameter to its attribute name. 
        For example, "x velocity (su s⁻¹): " is converted to "velocity.x_component" 
        """
             
        label_text = label_text.replace("selected body ", "")
        label_text = label_text.replace("?", "")
        label_text = label_text.replace(":", "")
        label_text = label_text.replace(" ", "_")
        
        # Removes the brackets indicating the parameter units from the label text
        if "(" in label_text:
            first_bracket_index = label_text.index("(")
            second_bracket_index = label_text.index(")")
            label_text = label_text[:first_bracket_index] + label_text[second_bracket_index+1:]
            
        # Remove any trailing underscores and makes the name lowercase
        label_text = label_text.rstrip("_")
        label_text = label_text.lower()
        
        # Finally some miscelannoeous modifications
        if label_text == "value_of_θ": label_text = "theta"
        elif label_text == "value_of_ε": label_text = "epsilon"
        # In this case the parameter is a vector so we need to convert it with a . (e.g. "x_coordinate" to "position.x_component")
        elif "angular" not in label_text and "calculation" not in label_text and ("velocity" in label_text or "coordinate" in label_text or "acceleration" in label_text or "momentum" in label_text):
            # If the label text has engine_ or total_ in front of it we remove it for now and then add it later
            prefix = ""
            if "engine_" in label_text: prefix = "engine_"
            if "total_" in label_text: prefix = "total_"
            
            label_text = label_text.replace("engine_", "").replace("total_", "").replace("coordinate", "position")
            
            if label_text[0] in ("x", "y"):
                vector_name, vector_property = label_text[2:], label_text[:2]+"component"
            else:
                vector_name, vector_property = label_text.split("_")
            
            label_text = prefix+vector_name+"."+vector_property
        
        return label_text      


    def translate_space_units(self, value: int|float, attribute_name: str, mode: Literal["from", "to"]):
        """This function translates a value for a certain attribute based on how it changes as the space unit changes.
        Some attributes are not modified (if they are not in self.ATTRIBUTES_WITH_SPACE). 
        You can specify if you want to convert FROM space units to meters or from meters TO space units.
        Remember that meters are the unit used for space in the Simulation, Body and Barnes Hut objects."""
        
        if attribute_name in self.ATTRIBUTES_WITH_SPACE:
            # Space units are translated by either multiplying or dividing the inputted value by a certain value
            match self.space_units:
                case "Kilometers": conversion_scalar = 1000
                case "Astronomical units (AU)": conversion_scalar = 149597870700
                case "Light seconds": conversion_scalar = 299792458
                case "Meters": conversion_scalar = 1
                
            # We raise the conversion scalar to the power of the order variable; a negative value means division.
            match mode:
                case "from": order = 1 # We multiply by the conversion scalar when converting from space units
                case "to": order = -1 # We divide by the conversion scalar when converting from space units
                
            if "kinetic_energy" in attribute_name or "angular_momentum" in attribute_name:
                order *= 2 # These attributes have a factor of su² so order of conversion must be doubled
            # coordinates are subject to centering offsets so we need to account for these
            elif attribute_name == "position.x_component":
                value += order*self.centered_body.position.x_component
            elif attribute_name == "position.y_component":
                value += order*self.centered_body.position.y_component
                
            value *= conversion_scalar**order # This finally translates the value.
        
        return value


    def process_parameter_input(self, object_name: str|None, attribute_name: str|None, inputted_value: str, graph_index: int|None = None) -> None:
        """This function checks a user's input for a certain attribute,
        and sets the attribute to their inputted value if it's correct input.
        It also updates the displayed value for that attribute."""

        valid_input, processed_input = self.check_valid_input(attribute_name, inputted_value)
                    
        if valid_input: # We now set the attribute if the input was valid
            if graph_index is None:
                if attribute_name == "id": # If the id of the selected body changes we must update self.contained_bodies
                    self.contained_bodies.pop(self.selected_body.id)
                    self.contained_bodies[processed_input] = self.selected_body
                exec(f"{object_name}.{attribute_name} = processed_input")
            else: # This means we are updating the time range value in self.graphs
                self.graphs[graph_index][3] = processed_input # Set the time range to the inputted value
                self.graphs[graph_index][2].set(processed_input) # Set the displayed time range value stringvar to the inputted value         
                           
        # We will now update the value of the attribute displayed to the actual value.
        # This will either display the user's input if it was valid or display their input being cleared if it was invalid
        if attribute_name not in (None, "body_to_orbit_around", "centered_body"):
            value_to_display = eval(f"{object_name}.{attribute_name}")
            value_to_display = self.translate_space_units(value_to_display, attribute_name, "to")
            if attribute_name not in ("is_immobile", "id"):
                value_to_display = f"{value_to_display:.3e}"        
            self.stringvars[attribute_name].set(str(value_to_display))
        elif graph_index is not None: # This sets the value for the time range shown to the actual value
            self.graphs[graph_index][2].set(self.graphs[graph_index][3])
         
         
    def check_valid_input(self, attribute_name: str, inputted_value: str):
        """This function checks that the input to an attribute was valid, and also processes the input."""
        
        gtez = ("theta", "epsilon", "arrow_scale_factor", "mass", "velocity.magnitude", # Attributes that should be greater than or equal to zero
                "engine_acceleration.magnitude", "momentum.magnitude", "kinetic_energy")
        gtz = ("time_step", "frames_per_second", "radius", "semi_major_axis_length", None) # Attributes that should be greater than zero
                  
        valid_input = True # We start from the assumption our user inputted correctly, then check if that's actually true if needed
        
        # This block of code is used to process the user input and/or see if it is valid
        # It does this by checking the input in different ways based on attribute_name
        match attribute_name:
            case "is_immobile": 
                match inputted_value.lower(): # The input must be a string of either "True" or "False"
                    case "true": inputted_value = True
                    case "false": inputted_value = False
                    case _: valid_input = False
                    
            case "id":
                lowercase_ids = [id.lower() for id in self.contained_bodies]
                existing_ids = ("origin", "center of mass", *lowercase_ids)
                valid_input = not (inputted_value.lower() in existing_ids) # The inputted value must not be an already existing id
                valid_input = valid_input and inputted_value != "" # The inputted value must be a non-empty string
                
            case "body_to_orbit_around" | "centered_body": # In this case we translate the user's selection to a Body object
                match inputted_value:
                    case "Origin":
                        inputted_value = Body("Origin", 0, 0, Vector(0, 0))
                    case "Center of mass": 
                        inputted_value = self.find_center_of_mass()
                    case _ if inputted_value in self.contained_bodies:
                        inputted_value = self.contained_bodies[inputted_value]
                    case _:
                        inputted_value = None
                                            
                if attribute_name == "centered_body" and inputted_value.id != self.centered_body.id: # The centered body has changed so we clear all trail points
                    self.trail_points = []
                    
            case _: # This case means the input is simply numerical
                try:
                    inputted_value = float(inputted_value)
                    inputted_value = self.translate_space_units(inputted_value, attribute_name, "from")            
                    valid_input = not ((attribute_name in gtez and inputted_value < 0) or (attribute_name in gtz and inputted_value <= 0))
                except ValueError: # The user has inputted a value which cannot be converted to float and is therefore invalid
                    valid_input = False
                    
        return valid_input, inputted_value
         

    def set_dropdown_menus(self) -> None:
        """This code refreshes the two dropdown menus in the user interface (the one for the centered body and the body to orbit around).
        It sets the values that can be selected and also performs any checks needed."""
        
        # Find the dropdown options for the centered body dropdown menu and apply them
        dropdown_options = ["Origin"]
        if len(self.contained_bodies) > 0: # We only give the center of mass option if there is at least 1 body
            dropdown_options.append("Center of mass")
            body_ids = list(self.contained_bodies.keys())
            dropdown_options.extend(body_ids)
        self.dropdowns["Centered body: "].configure(values=dropdown_options) # Sets the dropdown options of the centered body dropdown to dropdown_options
        
        # If the currently centered body does not exist (e.g. it was merged or deleted by the user), we set the centered body to the origin.
        if self.dropdowns["Centered body: "].get() not in ("Origin", "Center of mass", *self.contained_bodies.keys()):
            self.dropdowns["Centered body: "].current(0) # Sets the dropdown menu selection to the first option, in this case the origin
            self.centered_body = Body("Origin", 0, 0, Vector(0, 0))

        # Find the dropdown options for the body to orbit around (i.e. all bodies except the selected body).            
        if self.selected_body is None or self.selected_body.id not in self.contained_bodies:
            # If the selected body does not exist due to merging or deletion, then ther are no dropdown options and we clear the selected body
            self.selected_body = None
            dropdown_options = []
        else:
            # This is a list comprehension, which simply gets the id of each body in the simulation except for the selected body
            dropdown_options = [id for id in self.contained_bodies if id != self.selected_body.id]
            
        self.dropdowns["Body to orbit around: "].configure(values=dropdown_options) # Sets the dropdown options of the body to orbit around dropdown to dropdown_options
        
        # Clears the selection if the body to orbit around chosen does not exist, again either due to merging or deletion by the user
        if self.dropdowns["Body to orbit around: "].get() not in self.contained_bodies:
            self.dropdowns["Body to orbit around: "].selection_clear()


    def set_selected_body_fields(self):
        """This function sets the stringvar objects responsible for displaying the properties of a body."""
        
        # A list of all the attributes displayed about the selected body
        selected_body_attributes = (
            "id", "is_immobile", "position.x_component", "position.y_component", "mass", "radius",
            "velocity.direction", "velocity.magnitude", "velocity.x_component", "velocity.y_component", 
            "acceleration.direction", "acceleration.magnitude", "acceleration.x_component", "angular_momentum", 
            "acceleration.y_component", "engine_acceleration.direction", "engine_acceleration.magnitude", 
            "engine_acceleration.x_component", "engine_acceleration.y_component", "momentum.magnitude", 
            "momentum.x_component", "momentum.y_component", "kinetic_energy", "angular_velocity")
        
        for attribute_name in selected_body_attributes:
            
            # If the user is currently inputting a value for that attribute, we do not set the stringvar (as this would clear their input)
            try:
                focussed_widget = self.master.nametowidget(self.master.call("focus").__str__()) # The name of the widget the user is focussed on
                
                # If the widget is an entry widget (used to display body properties) and the stringvar of that entry is the same as the current attribute's we skip this iteration
                if isinstance(focussed_widget, Entry) and focussed_widget.cget("textvariable") == self.stringvars[attribute_name].__str__():
                    continue
            except KeyError: # A bug in tkinter sometimes happens which causes a KeyError, so it is unfortunately necessary to use a try/except block.
                pass

            if self.selected_body is None: # Values are displayed as "-" if there is no selected body
                value = "-"
            else:
                # Retrieve the value stored
                match attribute_name:
                    case "angular_velocity":
                        value = self.selected_body.angular_velocity(self.centered_body, self.time_step)
                    case "angular_momentum":
                        value = self.selected_body.angular_momentum(self.centered_body)
                    case _:
                        value = eval(f"self.selected_body.{attribute_name}")
                
                value = self.translate_space_units(value, attribute_name, "to") # Any space units retrieved are in meters so we need to convert these
                        
                # Translate any numerical values into 3-decimal standard form (e.g. 3.56 * 10^3)  
                if attribute_name not in ("id", "is_immobile"):
                    value = f"{value:.3e}"
                    
            self.stringvars[attribute_name].set(str(value)) # Sets the stringvar (i.e. the displayed value in the textbox) to the value of the attribute


    def set_graphs(self):
        """This function sets the graphs widgets; this means updating the values displayed in the textboxes for each graph, 
        setting the data points that need to be drawn, and drawing the graph."""
        
        for index, graph_info in enumerate(self.graphs):
            # Remember that self.graphs contains lists containing the below variables.
            canvas, current_value_stringvar, time_range_stringvar, time_range, data_points, attribute_name = graph_info
                 
            # Checks if the attribute to be plotted is set to nothing or a selected body property is selected but there is no selected body                   
            if attribute_name is None or ("total" not in attribute_name and self.selected_body is None):
                # if so, no value can be displayed and no data points either, so we clear both
                current_value_stringvar.set("-")
                data_points = []
                
            else:
                if "total" in attribute_name: # The attribute is a total so pygame_object.total_of_property() should be used
                    attribute_name = attribute_name.replace("total_", "")
                    displayed_value = self.total_of_property(attribute_name, self.centered_body)
                else: # The attribute is an attribute of the selected body
                    match attribute_name:
                        case "angular_velocity":
                            displayed_value = self.selected_body.angular_velocity(self.centered_body, self.time_step)
                        case "angular_momentum":
                            displayed_value = self.selected_body.angular_momentum(self.centered_body)
                        case _:
                            displayed_value = eval(f"self.selected_body.{attribute_name}")
                    
                displayed_value = self.translate_space_units(displayed_value, attribute_name, "to")
                current_value_stringvar.set(f"{displayed_value:.3e}") # Displays the value in 3 significant figure standard form
                
                if not self.paused: # Data points are only added and the graph is only drawn if the simulation isn't paused
                    if len(data_points) == 0 or data_points[-1][0] != self.time_elapsed: # Add a new data point only if simulation time has passed
                        data_points.append((self.time_elapsed, displayed_value)) # Remember data points is a list of tuples 
                        
                    # Remove data points that are too far in the past.
                    number_to_remove = max(0, int(len(data_points)-self.frames_per_second*time_range))
                    del data_points[:number_to_remove]
                
            self.draw_graph(canvas, data_points)             
            self.graphs[index][4] = data_points # Set the data points stored in self.graphs to the newly updated ones
                            
            
    def draw_graph(self, canvas: Canvas, data_points) -> None:
        """This function draws a graph on a given tkinter Canvas object for the given data points."""
        
        canvas.delete("data_points") # Clear the previously drawn data points
        canvas.delete("warning") # Delete the floating point warning text from before
        
        if len(data_points) >= 2:
            
            # First we find the maximum and minimum x and y values for the data points
            min_x = min([x for x, y in data_points])
            max_x = max([x for x, y in data_points])
            min_y = min([y for x, y in data_points])
            max_y = max([y for x, y in data_points])
  
            # We then use these values to calculate a width scale factor and offset for the data points.
            # This will scale the data points to fit inside of the 100x150 pixel area inside of the canvas we draw them             
            width_scaling = 150/(max_x-min_x)
            width_offset = 25-min_x*width_scaling
            
            height_scaling = 100/(max_y-min_y+1e-31)
            height_offset = 25-min_y*height_scaling
            
            # We check for a very small change from the previous to current value.
            # If so, we warn the user there may be a floating point error.
            # For example, when viewing graphs of properties that should stay the same they may fluctuate very slightly due to floating point error 
            last_y, second_to_last_y = data_points[-1][1], data_points[-2][1]
            if last_y != 0 and abs(second_to_last_y-last_y)/last_y < 0.00001: # We check for both a small absolute and relative change
                canvas.create_text(100, 10, text="Floating point error warning", tags="warning", fill="red")
            
            # We transform the points using the scaling factors and offsets and draw the data points onto the graph connected by straight lines
            drawn_data_points: list[tuple[int, int]] = []
            for x, y in data_points:
                drawn_data_points.append((x*width_scaling+width_offset, 150-(y*height_scaling+height_offset)))
                            
            canvas.create_line(drawn_data_points, tags="data_points", fill="black")


    def main_loop(self) -> None:
        """This is the main loop of the program which repeatedly executes until 
        the user quits the program."""  
        
        # We run the pygame main loop in a separate thread
        # This is because if we didn't, the user interface would update at the same rate as the bodies move
        # This is problematic because if frames_per_second is set to a low value the user interface only responds 1 time per second
        threading.Thread(target=self.pygame_main_loop).start()
                
        while True:
            self.master.update()
            self.set_dropdown_menus()
            self.set_selected_body_fields()
            self.set_graphs()
            self.stringvars["time_elapsed"].set(self.time_elapsed)
                        
            self.draw_pygame_window()
            self.check_events()
            pygame.display.update()
        
    # These next functions return a widget for each individual window of the user interface
                        
    def buttons_window(self, parent: tkinter.Widget) -> None:
        """This function constructs a Frame object containing the "Basic buttons" section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        Button(container, text="Quit", command=lambda: os.kill(os.getpid(), 9)).pack()
        Button(container, text="Un/Pause", command=lambda: setattr(self, "paused", not self.paused)).pack()
        Button(container, text="Step", command=self.step).pack()
        Button(container, text="Add body", command=lambda: setattr(self, "adding_body_mode", not self.adding_body_mode)).pack()
        Button(container, text="Delete selected body", command=lambda: self.remove_body()).pack()
        Button(container, text="Un/Show trail on selected body", command=lambda: setattr(self, "show_trail", not self.show_trail)).pack()
                
        return container 


    def parameters_window(self, parent: tkinter.Widget) -> None:
        """This function constructs a Frame object containing the "Simulatin parameters" section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        # This is a tuple of tuples providing information about parameters displayed using a textbox in this page
        # It contains, the label text and the object the attribute is stored in
        textbox_parameter_info = (
            ("Time elapsed (s): ", "self"), 
            ("Value of θ (>= 0): ", "self"),
            ("Value of ε (su, >= 0): ", "self"),
            ("Time step (> 0): ", "self"),
            ("Arrow scale factor (>= 0): ", "self"),
            ("Frames per second (> 0): ", "self"))
        
        for index, parameter_info in enumerate(textbox_parameter_info):
            attribute_label, associated_object = parameter_info
            attribute_name = self.label_to_attribute_name(attribute_label)
            
            label = Label(container, text=attribute_label)
            label.grid(row=index, sticky="w")
            
            textbox = Entry(container, relief="solid", width=20, textvariable=self.stringvars[attribute_name])
            
            if attribute_name == "time_elapsed":
                textbox.configure(state="disabled") # Disable editing textbox as time elapsed is a non-editable property
                
            # Textbox bindings perform a command when an event happens with the textbox.
            # <Return> means the enter button was clicked, <FocusOut> means the user clicked away from that textbox
            textbox.bind("<Return>", lambda event, obj=associated_object, attr=attribute_name: self.process_parameter_input(obj, attr, self.stringvars[attr].get()))
            textbox.bind("<FocusOut>", lambda event, obj=associated_object, attr=attribute_name: self.stringvars[attr].set(f"{eval(f"{obj}.{attr}"):.3e}"))
            
            textbox.grid(row=index, sticky="w", padx=label.winfo_reqwidth()) # .grid() is a way of adding widgets where they are split into a row-column layout
            
        # This is a tuple of tuples providing information about the parameters displayed in drop down menus in this page.
        # It contains, in order, the label text, the object the attribute is stored in and the dropdown options.
        dropdown_parameter_info = (
            ("Centered body: ", "self", ("Origin")),
            ("Space units (su): ", "self", ("Meters", "Kilometers", "Light seconds", "Astronomical units (AU)")),
            ("Acceleration calculation: ", "self", ("Direct Sum", "Barnes-Hut")),
            ("Collision detection: ", "self", ("Direct Sum (Slow)", "Direct Sum (Optimized)", "Barnes-Hut", "None")))
        
        for index, parameter_info in enumerate(dropdown_parameter_info):
            attribute_label, associated_object, dropdown_options = parameter_info
            attribute_name = self.label_to_attribute_name(attribute_label)
            
            label = Label(container, text=attribute_label)
            label.grid(row=index+len(textbox_parameter_info), sticky="w")
            
            dropdown_menu = Combobox(container, width=20, values=dropdown_options, state="readonly")
            dropdown_menu.current(0)
            
            # The <<ComboboxSelected>> event happens when the user clicks on one of the dropdown menu options
            if index >= 1:
                dropdown_menu.bind("<<ComboboxSelected>>", lambda event, widget=dropdown_menu, obj=eval(associated_object), attr=attribute_name: setattr(obj, attr, widget.get()))
            else: # The centered body drodown menu options change and also need to be converted to Body objects, so we must treat it differently from the other dropdown menus
                self.dropdowns["Centered body: "] = dropdown_menu
                dropdown_menu.bind("<<ComboboxSelected>>", lambda event, widget=dropdown_menu, obj=associated_object, attr=attribute_name: self.process_parameter_input(obj, attr, widget.get()))
                
            dropdown_menu.grid(row=index+len(textbox_parameter_info), sticky="w", padx=label.winfo_reqwidth())
                                
        return container
        
        
    def selected_body_info_window(self, parent: tkinter.Widget) -> None:
        """This function constructs a Frame object containing the "Information" part of the "Selected body menu" 
        section of the simulation interface and returns it"""
        
        container = Frame(parent)
            
        scrollbar, container_background, textbox_container = self.scrollable_frame(container, width=375, height=400)
        
        selected_body_attribute_labels = (
            "id: ", "is immobile?: ", "x coordinate (su): ", "y coordinate (su): ",
            "mass (kg, >= 0): ", "radius (su, > 0): ", "velocity direction (rad): ", 
            "velocity magnitude (su s⁻¹, >= 0): ", "x velocity (su s⁻¹): ", 
            "y velocity (su s⁻¹): ", "acceleration direction (rad): ",
            "acceleration magnitude (su s⁻², >= 0): ", "x acceleration (su s⁻²): ", 
            "y acceleration (su s⁻²): ", "engine acceleration direction (rad): ", 
            "engine acceleration magnitude (su s⁻², >= 0): ", 
            "engine x acceleration (su s⁻²): ", "engine y acceleration (su s⁻²): ", 
            "x momentum (kg su s⁻¹): ", "y momentum (kg su s⁻¹): ", 
            "momentum magnitude (kg su s⁻¹, >= 0): ", "kinetic energy (kg su² s⁻², >= 0): ", 
            "angular velocity (rad s⁻¹): ", "angular momentum (kg su² s⁻¹): ")
                
        for index, attribute_label in enumerate(selected_body_attribute_labels):
            attribute_name = self.label_to_attribute_name(attribute_label)
            
            # We not create the label and textbox for this attribute
            label = Label(textbox_container, text=attribute_label)
            label.grid(row=index+1, sticky="w")
            
            textbox = Entry(textbox_container, relief="solid", width=20, textvariable=self.stringvars[attribute_name])
            
            # Disable editing the textbox if it is displaying a non-editable property
            if attribute_label in ("angular velocity (rad s⁻¹): ", "angular momentum (kg rad s⁻¹): ", "acceleration magnitude (su s⁻², >= 0): ", 
                                  "x acceleration (su s⁻²): ", "y acceleration (su s⁻²): ", "acceleration direction (rad): "):
                textbox.configure(state="disabled")
                
            textbox.bind("<ButtonPress-1>", lambda event: setattr(self, "paused", True)) # if the user clicks on the textbox the simulation pauses
            textbox.bind("<Return>", lambda event, attr=attribute_name: self.process_parameter_input("self.selected_body", attr, self.stringvars[attr].get()))
            
            textbox.grid(row=index+1, sticky="w", padx=label.winfo_reqwidth())
            
        scrollbar.pack(side="right", fill="y")
        container_background.pack(anchor="nw")
    
        return container
                
                
    def selected_body_orbit_generator_window(self, parent: tkinter.Widget) -> None:
        """This function constructs a Frame object containing the "Orbit generator" part of the "Selected body menu" 
        section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        Label(container, text="Body to orbit around: ").grid(row=1, sticky="w")
        
        # Create the "body to orbit around" dropdown menu and add it to to the self.dropdowns attribute so it can be updated later
        # We bind it to self.check_parameter_input, since a user will choose a body id which will need to be converted to a Body object
        dropdown_menu = Combobox(container, width=14)
        self.dropdowns["Body to orbit around: "] = dropdown_menu
        dropdown_menu.bind("<<ComboboxSelected>>", lambda event, widget=dropdown_menu: self.process_parameter_input("self", "body_to_orbit_around", widget.get()))
        dropdown_menu.grid(row=1, sticky="w", padx=120)
        
        Label(container, text="Semi-major axis length (su, > 0): ").grid(row=2, sticky="w")
        
        # We not create the semi major axis length textbox, and give it bindings when the user presses enter and when they click something else.
        textbox = Entry(container, width=11, textvariable=self.stringvars["semi_major_axis_length"])
        textbox.bind("<Return>", lambda event: self.process_parameter_input("self", "semi_major_axis_length", self.stringvars["semi_major_axis_length"].get()))
        textbox.bind("<FocusOut>", lambda event: self.stringvars["semi_major_axis_length"].set(f"{self.translate_space_units(getattr(self, "semi_major_axis_length"), "semi_major_axis_length", "to"):.3e}"))
        textbox.grid(row=2, sticky="w", padx=180)
        
        # Finally, a button that when clicked, uses the inputs from the above widgets to generate a stable orbit for the selected body.
        Button(container, text="Generate stable orbit", command=self.generate_stable_orbit).grid(sticky="w", padx=60)

        return container        
        

    def graphs_window(self, parent: tkinter.Widget) -> None:
        """This function constructs a Frame object containing the "Graphs" 
        section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        scrollbar, container_background, graphs_container = self.scrollable_frame(container, width=210, height=400)
        
        # Contains all of the labels of parameters the user can choose to graph
        graph_option_labels = (
            "nothing", "selected body x coordinate", "selected body y coordinate",
            "selected body velocity direction", "selected body velocity magnitude",
            "selected body x velocity", "selected body y velocity",
            "selected body acceleration direction", "selected body acceleration magnitude",
            "selected body x acceleration", "selected body y acceleration",
            "selected body momentum magnitude", "selected body x momentum",
            "selected body y momentum", "selected body kinetic energy",
            "selected body angular velocity" , "selected body angular momentum",
            "total x velocity", "total y velocity", 
            "total velocity magnitude", "total x acceleration",
            "total y acceleration", "total acceleration magnitude)",
            "total x momentum", "total y momentum",
            "total momentum magnitude", "total kinetic energy",
            "total angular velocity", "total angular momentum")
        
        # We now add 10 graph boxes, each containing a graph and two textboxes
        for i in range(10):
            graph_frame = Frame(graphs_container)
            
            # We now create the grpah canvas and draw the graph axes and labels, which are permenant
            canvas = Canvas(graph_frame, width=200, height=150, background="white")
            canvas.create_line(25, 130, 180, 130, fill="black") # Draw the x axis
            canvas.create_line(25, 20, 25, 130, fill="black") # Draw the y axis
            canvas.create_text(100, 140, text="Real time (s)") # Draw the x axis label
            canvas.create_text(10, 75, text="Parameter value", angle=90) # Draw the y axis label
            canvas.grid(sticky="w", row=0)
            
            label = Label(graph_frame, text="Selected parameter: ")
            label.grid(row=1, column=0, sticky="w")
            
            # We now add the dropdown menu from which the user can choose the option they are graphing
            # WHen an option is clicked, the data points are cleared and the attribute name stored in self.graphs is set to the one selected
            dropdown_menu = Combobox(graph_frame, values=graph_option_labels, width=11, state="readonly")
            dropdown_menu.current(0)
            self.style.configure("TCombobox", postoffset=(0, 0, 150, 0))
            # This binding changes self.graphs so that the data points are cleared and the attribute name is set to the newly selected one
            dropdown_menu.bind("<<ComboboxSelected>>", lambda event, index=i, widget=dropdown_menu: self.graphs.__setitem__(index, self.graphs[index][:4]+[[], self.label_to_attribute_name(widget.get())]))
            dropdown_menu.grid(row=1, sticky="w", padx=label.winfo_reqwidth())
            
            # We now add the textbox showing the current value of the parameeter being graphed
            Label(graph_frame, text="Current value: ").grid(row=2, sticky="w")
            current_value_stringvar = StringVar(graph_frame, "-")
            current_value_entry = Entry(graph_frame, width=20, textvariable=current_value_stringvar, state="disabled")
            current_value_entry.grid(row=2, sticky="w", padx=80)
            
            # We now add the textbox showing how far in the past the value is graphed for
            # The initial time range is 5 seconds
            Label(graph_frame, text="Time range (s): ").grid(row=3, sticky="w")
            time_range_stringvar = StringVar(graph_frame, "5")
            current_value_entry = Entry(graph_frame, width=19, textvariable=time_range_stringvar)
            current_value_entry.bind("<Return>", lambda event, index=i: self.process_parameter_input(None, None, self.graphs[index][2].get(), index))
            current_value_entry.bind("<FocusOut>", lambda event, index=i, stringvar=time_range_stringvar: stringvar.set(self.graphs[index][3]))
            current_value_entry.grid(row=3, sticky="w", padx=85)
            
            graph_frame.pack()
            self.graphs.append([canvas, current_value_stringvar, time_range_stringvar, 5, [], None])
                        
        scrollbar.pack(side="right", fill="y")
        container_background.pack()
                
        return container
    
    
    def manual_window(self, parent: tkinter.Widget) -> None:
        """This function constructs a Frame object containing the "Manual" 
        section of the simulation interface and returns it"""
        
        container = Frame(parent)
        
        # text_mapping contains a mapping that converts topics to text describing them.
        # This is then used to show the user information.
        text_mapping = {
            "How to use the interface": 
                "", 
            "Using textboxes": 
                "To input a value, simply click on the textbox and type your desired value into it. "
                "To the left of a textbox is in brackets, the unit of the parameter and any condition (e.g. >= 0). "
                "For numerical inputs, 3 decimal place standard form is used. ae+b means a × 10ᵇ and ae-b means a × 10⁻ᵇ. "
                "When you press enter, your input is processed. If it is invalid, it will be cleared. "
                "When you click away from a textbox, its value will be resetset to the attribute it is displaying. "
                "Some textboxes are grayed out, meaning their values cannot be edited. ",
            "Adding a body": 
                "To add a body, click on the 'Add body' button in the basic buttons section on the top-left. "
                "Then click on the screen where you want to add your body. You will be shown a preview. "
                "Once your body has been added, it will automatically be selected. ", 
            "Orbit generation": 
                "To generate an orbit, first select a body (it will be the orbiting body). "
                "Then select a body to orbit around and input a semi major axis length. "
                "The major axis length is the length of the longest line that can be drawn in the ellipse, "
                "and the semi-major axis length is half that. "
                "Next, you will be shown an orbit preview. "
                "If your input for the axis length is too short, no preview will be displayed. "
                "Press the button to generate the orbit. "
                "This is ignored if the input for the axis length is too small.",
            "Using the graphs": 
                "Graphs have arbitrary axes scaling for the y axis (the axes just expand to graph all the points). "
                "The time range is the amount of time in the past for which the parameter is plotted.", 
            "Selecting a body": 
                "To select a body, simple click on it. It will be highlighted in yellow. "
                "Its velocity (red), acceleration (purple) and engine acceleration (green) will be shown as arrows. ", 
            "How the simulation works": "", 
            "Acceleration calculation": 
                "Acceleration calculation is either done using the Barnes-Hut algorithm or Direct Sum algorithm. "
                "The Direct Sum algorithm calculates the EXACT total acceleration on each body by going through "
                "each pair of bodies and calculating the force due to gravity between them. "
                "Note that this takes exponentially (squared) more time as the number of bodies increases. "
                "As an alternative, the Barnes-Hut algorithm APPROXIMATES the acceleration on each body. "
                "Let's say it's finding the acceleration on a body X. This algorithm clusters bodies "
                "far away from X and approximates the acceleration on X due to those bodies using the "
                "center of mass and total mass of the cluster. "
                "This is MUCH faster than the Direct Sum approach for a large number of bodies.",
            "Collision detection": 
                "Collision detection can either be disabled, Direct Sum (Slow or optimized), or Barnes-Hut. "
                "Please read the acceleration calculation page for an explanation of Barnes-Hut and Direct-Sum. "
                "When applied to collision detection, Direct Sum simply checks for a collision between each pair of bodies. "
                "Meanwhile, Barnes-Hut is much faster as it excludes bodies that are too far away for collision "
                "when finding the collisions on a body. The Direct Sum collision detection method still grows "
                "in time exponentially (even when using the optimized version), and the Barnes-Hut version is again "
                "much faster.",
            "How bodies move": 
                "Bodies are moved using the 'Euler method'. Each step or frame in the simulation, "
                 "A body's velocity is incremented by its acceleration multiplied by the 'Time step' parameter. "
                "Then, its position is incremented by its velocity again multiplied by the time step. "
                "In more mathematical terms, 1. velocity = velocity + (acceleration due to gravity + engine acceleration) × time step. "
                "2. position = position + velocity × time step. A smaller time step value means a more accurate but slower simulation. "
                "The opposite is also true.",
            "Simulation parameters": 
                "",
            "The θ parameter": 
                "θ is the parameter used for the Barnes-Hut algorithm (acceleration calculation only). "
                "Please read the 'Acceleration calculation' section of 'How does the simulation work' first. "
                "A higher value of θ means that the acceleration on each body is approximated more, "
                "by decreasing the distance at which clustering happens (clustering is more accurate further away). "
                "This means a faster but less accurate simulation. The opposite is also true. ",
            "The ε parameter": 
                "ε is the parameter used in a technique called 'smoothening'. "
                "Sometimes two bodies get too close and their acceleration spikes unnaturally. "
                "A higher value of ε reduces this effect, by setting the minimum distance we consider two bodies to have.",
            "Time step": 
                "Please see the 'How bodies move' section in the 'How the simulation works' section.",
            "Arrow length factor": 
                "The length of vector arrows drawn on the selected body are multiplied by this value.",
            "Frames per second": 
                "This is how many times per second the Euler method is used. "
                "See the 'How bodies move' section in the 'How the simulation works' section for what the Euler method is.\n",
            "Centered object": 
                "This is the object placed at the center of the simulation. "
                "This can be either the origin, the center of mass of the simulation or a body. ",
            "Space units (su)": 
                "This changes the units for space used in the simulation. "
                "Properties like coordinates, velocities and so on are scaled to be in the right units. "
                "For example, if the space unit selected is light seconds, then coordinates will be in light seconds. ",
            "Selected body id": 
                "This is a unique identifier given to every body. This means that two id's cannot be the same. "
                "An id is also not allowed to be either 'Center of mass' or 'Origin'.",
            "Angular properties": 
                "Angular properties (angular velocity or momentum) are calculated using the centered object of the simulation. "
                "Angular velocity is calculated by finding the change in angle from the centered body to the selected body divided by the time step. "
                "Angular momentum is calculated using the cross product of the vector from the centered body to the selected body and the selected body's velocity. "
        }
                
        # We now add newlines (\n) to limit the maximum length of the explanation on the screen
        maximum_length = 36 # Maximum length of a line in characters
        for topic, explanation in text_mapping.items():
            modified_explanation = ""
            words = explanation.split(" ")
            
            current_line = "" # Represents the current line of text we are processing
            for word in words:
                if len(current_line+word) > maximum_length: # This means we have reached our character limit, and should make a new line
                    modified_explanation += current_line + "\n"
                    current_line = word + " "
                else: # Otherwise we simply add the word to the current line
                    current_line += word + " "
            modified_explanation += current_line + "\n"
                    
            text_mapping[topic] = modified_explanation
                
        current_text_displayed = StringVar(container, "")
        text_widget = Label(container, textvariable=current_text_displayed, justify="left")
        
        navigation_interface = Treeview(container, height=20, show="tree") # Displays a hierarchical view of text
        
        # This contains tuples of tuples, each containing first an id and then a parent.
        # The id uniquely identifies that piece of text, and the parent is the id of the place the text is under.
        # An empty id means that it has no parent.
        categories_info: tuple[tuple[str, str]] = (
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
        
        for index, (id_, parent) in enumerate(categories_info): # We use id_ instead of id since id() is a builtin python function
            navigation_interface.insert(parent, index, id_, text=id_)

        navigation_interface.bind("<<TreeviewSelect>>", lambda event: current_text_displayed.set(text_mapping[event.widget.focus()]))
        
        navigation_interface.pack(side="left")
        text_widget.pack(side="left")
                
        return container



# %%
TkinterWindow()

# Ask about if coding style is the way AQA likes it


