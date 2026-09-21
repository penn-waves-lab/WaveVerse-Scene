"""Scene-aware action prompt with unoccupied-floor motion endpoints."""

ACTION_DESIGN_PROMPT = r"""You are an experienced human action designer, who is expert in designing daily tasks of human within a given environment while considering the context given by the environment. Please assist me in drafting descriptions of daily human motions. You need to give a text description of tasks, including the description of the motion itself, the start and the end positions.
The environment we have is also generated from a given environment prompt. Please ensure that the task description is feasible within the given environment, like the action can be done by a person within the environment and the starting and ending points are in the environment.
Below is an example of an environment prompt, the details of the generated environment, and an examples of task descriptions which you should generate.
Note: the units for the coordinates are in meters.

For example:
Environment prompt:
generate a living room.
Environment details:
Floor plan:
living room | [(0, 0), (0, 6), (7, 6), (7, 0)]
Wallheight: 2.7
Doors:
door|0|exterior|living room | exterior | living room | [(2.08, 6), (4.08, 6)]
Windows:
window|wall|living room|south|3|0|0 | living room | [(5.27, 0), (6.75, 0)]
window|wall|living room|south|3|1|1 | living room | [(2.70, 0), (4.18, 0)]
window|wall|living room|south|3|2|2 | living room | [(0.27, 0), (1.75, 0)]
Floor objects:
sectional_sofa-0 (living room) | living room | [(5.89, 0, 2.84), (7.05, 0.72, 5.55)]
tv_stand-0 (living room) | living room | [(0, 0, 3.34), (0.54, 0.74, 5.06)]
bookshelf-0 (living room) | living room | [(6.51, 0, 0.13), (7.05, 1.92, 1.27)]
armchair-0 (living room) | living room | [(3.77, 0, 3.44), (4.62, 1.00, 4.96)]
Wall objects:
painting-0 (living room) | living room | [(6.97, 3.87), (7.00, 4.63)]
wall-mounted_shelf-0 (living room) | living room | [(4.15, 5.56), (4.95, 6.00)]
Small objects:
55 inch tv-0|tv_stand-0 (living room) | living room | [(0.24, 0.93, 3.80)]
coaster-0|side_table-0 (living room) | living room | [(4.49, 0.73, 5.80)]

Here are some guidelines for you to understand the above environment details:
1. The space is represented in a  X, Y, Z coorindate system, where Y represents the height. (0, 0, 0) denotes the bottom-left corner of the room.
2. Whenever there are only two numbers for a coordiante, it represents (X, Z), ommiting height Y. 
3. The detailed environment consists of six parts, Floor plan, Doors, Windows, Floor objects, Wall objects and Small objects.
4. The floor plan is represented as: room name | four coordinates of four corners.
5. Doors are represented as: door name | room 1 | room 2 | two coordinates of the projected doors on X-Z plane (line).
6. Windows are represented as: window name | room | two coordinates of the projected doors on X-Z plane (line).
7. Floor objects are represetned as: floor object name | room | two 3D coordinates which compose the 3D bounding box for the object.
8. Wall objects are represented as: wall object name | room | two 2D coordinates which compose the 2D bounding box for the projected object on X-Z plane.
10. The object category is included in the id, you can have some induction based on the name to understand the object better, like the height for wall objects, size for small objects.

Task description examples:
A person walks from the 'open floor west (living room)' to the 'open floor east (living room)', from position (2.00, 1.50) to position (4.00, 1.50).
A person sidesteps from the 'open floor south (living room)' to the 'open floor north (living room)', from position (1.50, 1.50) to position (1.50, 3.00).
A person waves in place from the 'open floor center (living room)' to the 'open floor center (living room)', from position (2.00, 1.50) to position (2.00, 1.50).


1. The generated task description must provide beginning and stop points on unoccupied floor in the environment. Use distinct points for travelling actions. For an action performed in place, use the same safe floor position for both endpoints; do not request travel across the room.
2. You should derive the splatial relations among all objects in the room.
3. You need to consider the space between objects to ensure that the task(path) is feasible and can be achieved wihtout moving objects for a person. In general, tasks with more open space are preferred.
4. Provide the 2D coordinates of these points on the X-Z plane. Choose open floor with at least 0.75 meters of clearance from walls and object footprints, including throughout the intended route. Never use an object center or a position on furniture or exercise equipment as a human endpoint. Use descriptive names such as 'open floor west (room name)' for these free-space locations. Leave additional room for motions with extended arms or legs.
5. Objects in the scene do not interact with humans. So do not generate tasks that require interaction with objects, like picking up or putting down objects.
6. There might be multiple rooms, you can design a task from one room to another room.
7. Choose recognizable everyday actions appropriate to this environment. Describe one concrete action per task in a short sentence, without combining several actions or telling a story. The floor is level: do not request stairs, climbing, flips, falls, or movement on furniture. Keep the action text separate from the quoted location labels. Different tasks should use different actions, while remaining physically plausible in the available space.
8. The generated descriptions must follow the format of the examples. First provide the motion description, then the names of the open-floor start and end positions and their coordinates. The location names should be in the format of 'open floor description (room name)' and the coordinates in the format of 'from position (x1, z1) to position (x2, z2)'.


Now, I need do design actions for the below prompts:
Environment prompt:
{environment_prompt}
Envrionment details:
{environment_details}

Now suppose you are an experienced human action designer, generate {task_number} possible tasks for the task description generation, which should be as diverse as possible. Strictly follow the format provided in the example. Your response should be direct and without additional text at the beginning or end.
"""
