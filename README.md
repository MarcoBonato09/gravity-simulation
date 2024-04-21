# A real-time gravity simulation, according to Newton's law of gravitation. Currently there are no collisions.
![image](https://github.com/MarcoBonato09/gravity-simulation/assets/99590461/e2764122-b758-45b3-bd0e-c1e6b40a6385)
![image](https://github.com/MarcoBonato09/gravity-simulation/assets/99590461/ceaedfb6-a680-44de-b891-2bf86be69d8b)

## The core principle of operation
The basic principle behind this is that each body is a circle, placed in a Cartesian coordinate grid around an arbitrary origin, that stores its velocity vector in the x and y directions. 
Each time step, the gravity between each pair of bodies is calculated using [Newton's law of gravitation](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation).
This is then resolved into x and y vectors and added to each body's velocity vectors. There is also a function to generate stable orbits using the [Vis Viva equation](https://en.wikipedia.org/wiki/Vis-viva_equation).

## Simulation parameters
The properties of each body can be varied at will: for example their mass, position, radius and so on. It is also possible to center the grid on a body.
You will also notice two important sliders at the top: the FPS slider and the timestep slider. The FPS slider is decides the number of simulation time steps per second 
the simulation is executing. This therefore means that, while you can crank it to the max, your computer may not be able to keep up and you may not see an improvement 
from 400 to 500 fps, for example.

## The timestep slider
This is a very important option! For example, it is possible to recreate the Earth and Moon to scale using this program, but you don't want to wait a month for the 
moon to orbit in the program, do you? The timestep slider scales velocity vectors. So basically, if your timestep is 2, all bodies will be considered to be moving twice as fast.
Basically, this means your simulation will run faster, BUT it will be less accurate and may cause the simulation to break if velocity vectors are overscaled.
