sciworld_solver_system_prompt: str = """
You are a helpful assistant to do scientific experiments in a text-based environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action". Remember that you can only output one "Action:" in per response.

The available actions are:
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device (e.g. activate stopwatch)
deactivate OBJ: deactivate a device (e.g. deactivate stopwatch)
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ on OBJ: use a device/item on another object
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container or location
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: wait for 10 steps
wait1: wait for 1 step

CRITICAL RULES:
1. ALWAYS use the EXACT object names as shown in observations. Do NOT rename or abbreviate objects (e.g. if you see "a steel block", use "steel block", NOT "brick" or "block").
2. "focus on OBJ" is crucial. Most tasks require you to focus on something at the end. Read the task description to see WHAT to focus on.
3. The task description tells you EXACTLY what to do. Read it very carefully and follow each instruction.
4. When an action returns "No known action" or fails, check that you are using the exact object name and correct action format.
5. Use "look around" FIRST when you enter a new room to see what objects are available.
6. Use "teleport to LOC" to move between rooms instantly.

TASK-SPECIFIC GUIDANCE:
- "Find X, focus on X, move to box": (a) find the object, (b) focus on OBJ, (c) pick up OBJ, (d) teleport to target room, (e) move OBJ to box.
- "Determine angle/friction/property": (a) go to the room with the experimental objects, (b) look around to see all objects, (c) use the experimental objects (e.g. move steel block to inclined plane, activate/deactivate stopwatch), (d) observe results, (e) repeat for comparison, (f) focus on the correct answer.
- "Boil/freeze/melt/change state": (a) find the substance, (b) find a heat/cold source, (c) move substance to heat/cold source or use appropriate device, (d) activate the device, (e) wait for the change, (f) focus on result.
- For inclined plane experiments: move the steel block to each inclined plane one at a time, observe which slides faster/slower, then focus on the correct plane.
"""


# Few-shot examples for ScienceWorld tasks
sciworld_few_shots = {
    "sciworld_example_0": """Observation: This room is called the kitchen. In it, you see: a substance called air, a cupboard. In the cupboard is: a glass cup, a metal pot, a tin cup. a counter. On the counter is: a bowl (containing a red apple, a banana, an orange, a potato). a fridge. In the fridge is: nothing. a glass jar (containing a substance called sodium chloride). a lighter. a oven, which is turned off. In the oven is: nothing. a painting. a sink, which is turned off. In the sink is: nothing. a stove, which is turned off. On the stove is: nothing. a table. On the table is: nothing. a thermometer. You also see: A door to the bathroom (that is open). A door to the hallway (that is open). A door to the outside (that is open).
Your task is to: Find a non-living thing. First, focus on the thing. Then, move it to the orange box in the bathroom.
Thought: I need to find a non-living thing. A glass cup is a non-living thing. I must first focus on it, then pick it up, teleport to the bathroom, and move it to the orange box.
Action: focus on glass cup
Observation: You focus on the glass cup.
Action: pick up glass cup
Observation: You move the glass cup to the inventory.
Action: teleport to bathroom
Observation: You teleport to the bathroom.
Action: move glass cup to orange box
Observation: You move the glass cup to the orange box.
""",
    "sciworld_example_1": """Observation: This room is called the workshop. In it, you see: a substance called air, an inclined plane B, an inclined plane H, a steel block, a stopwatch, which is deactivated. a table. On the table is: a battery, wires, light bulbs. You also see: A door to the hallway (that is open).
Your task is to: Your task is to determine which of the two inclined planes (B, H) has the steepest angle. After completing your experiment, focus on the inclined plane with the steepest angle.
Thought: I need to determine which inclined plane is steeper. I should slide the steel block down each plane and compare the results. Let me start with inclined plane B.
Action: move steel block to inclined plane B
Observation: You move the steel block to the inclined plane B. The steel block slides down inclined plane B in 3 seconds, over a distance of 0.93 meters.
Thought: The steel block took 3 seconds on plane B. Now let me try plane H.
Action: pick up steel block
Observation: You move the steel block to the inventory.
Action: move steel block to inclined plane H
Observation: You move the steel block to the inclined plane H. The steel block slides down inclined plane H in 2 seconds, over a distance of 0.93 meters.
Thought: Plane H took 2 seconds (faster slide = steeper angle), plane B took 3 seconds. So plane H has the steepest angle.
Action: focus on inclined plane H
""",
    "sciworld_example_2": """Observation: This room is called the hallway. In it, you see: a substance called air, a picture. You also see: A door to the art studio (that is open). A door to the bedroom (that is open). A door to the greenhouse (that is open). A door to the kitchen (that is open). A door to the living room (that is open). A door to the workshop (that is open).
Your task is to: Find a living thing. Focus on the living thing.
Thought: I need to find a living thing. The greenhouse likely has plants. Let me teleport there.
Action: teleport to greenhouse
Observation: This room is called the greenhouse. In it, you see: a flower pot (containing soil and a peach tree). a jug (containing a substance called water). You also see: A door to the hallway (that is open). A door to the outside (that is open).
Thought: A peach tree is a living thing. I must focus on it to complete the task.
Action: focus on peach tree
""",
}
