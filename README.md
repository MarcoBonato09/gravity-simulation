# A real-time gravity simulation, according to Newton's law of gravitation
![image](https://github.com/user-attachments/assets/74c7901a-b245-4937-bd4b-429b6b45e556)


## The core principle of operation
The basic principle behind this is that each body is a circle, placed in a Cartesian coordinate grid around an arbitrary origin, that stores its velocity vector in the x and y directions. 
Each time step, the gravity between each pair of bodies is calculated using [Newton's law of gravitation](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation).
This is then resolved into x and y vectors and added to each body's velocity vectors. There is also a function to generate stable orbits using the [Vis Viva equation](https://en.wikipedia.org/wiki/Vis-viva_equation).

## Barnes-Hut
The program allows you to use the Barnes-Hut algorithm, which can either approximatd the forces due to gravity on a body or speed up collision detection. Note that for collision detection, Barnes-Hut does not approximate, so it is simply just a better algorithm than the brute force approach.

## The manual for more information
Please try downloading the python file for yourself and looking at the manual! This will give you all the other information you need.
