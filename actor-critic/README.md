# Running actor-critic for watermaze using place cell encodings

## Class descriptions

### watermaze
Describes the environment:
* A circular environment defined pool_radius, platform_radius, platform_location and stepsize (velocity of mouse)
* **momentum** makes the movements more realistic, similar to original rat movements thus restricting sudden change of directions
* The RMW (Reference Memory) and DMP (Delayed matching-to-place) task classes specify the repsective change in goal locations (RMW --> after 70% days of training, DMP --> everyday)

Important considerations:
* Set the **env.t=0** while reseting the environment.
* Terminates when either *timeup()* or *atgoal()* is True
* *move()* function interacts with the environment, allowing the agent to take an action in the environment

Todo:
* [ ] Write a proper *reset()* function
* [ ] Write a *step()* function to allow this env to be used like a gym environment

### PlaceCellEncoding
The feature encoding class using set number of place cells. Place cell centers are uniformly distributed in the radial and angular directions. *plotPlaceCells()* plots the centers of Place fields in the environment. *plotPlaceCellActivations()* creates a visualization (possibly an animation when called in a loop) of place cell activation values as the agent moves in an environment.

Todo:
* [ ] Make a super class for feature encoding and PlaceCellEncoding, GridCellEncoding, TileCoding etc. can be subclasses

### ActorCritic
Implements the actor-critic algorithm using TD-error. The action is sampled from the softmax of the actor output. *plot_critic_weights()* and *plot_actor_preferences()* can be used to visualize the actor and critic network preferred action/weights respectively. *TD_lambda()* implements the TD-error based training algorithm using eligibility traces.

## How to run
* ~~To visualize the updates and trajectories taken in each episode, run *actorCritic.py*: *python actorCritic.py* and uncomment line 228 in the *main* function (effectively use verbose=True for *TD_lambda()* function~~
* ~~To plot the learning performance (average number of steps taken every 500 episodes), run *python actorCritic.py* as it is. ~~
* To visualize the updates and trajectories taken in each episode, run *actorCritic_goalCode.py*: *python actorCritic_goalCode.py* and uncomment line 280 in the *main* function (effectively use verbose=True for *TD_lambda()* function
* To plot the learning performance (average number of steps taken every 200 episodes), run *python actorCritic_goalCode.py* as it is. 

-------------
## Results
* 10 days of navigation with 3 sessions on each day.
* Each session included 200 episodes, results averaged over 200 episodes (random start locations in each episode)
* Error bars indicate SEM
* Platform (goal) location changed to opposite quadrant on day 8
* Both place and tile coding agents perform equally well, but are unable to adjust to the shifted platform location. Tile coding agent fares slightly better at adjusting to new platform location
* Adding the goal feature encoding to the agent allows it to adjust better to new location --> holds true for both place and tile agents (tile coding agent again adjusts slightly quicker)

<img src="navigation%20result%20plots/RMW_tile.png" width="70%">
<img src="navigation%20result%20plots/RMW_place.png" width="70%">
<img src="navigation%20result%20plots/RMW_tile_goal.png" width="70%">
<img src="navigation%20result%20plots/RMW_place_goal.png" width="70%">
