# Running actor-critic for watermaze using place cell encodings

## Class descriptions

### watermaze
Describes the environment:
* A circular environment defined pool_radius, platform_radius, platform_location and stepsize (velocity of mouse)
* **momentum** makes the movements more realistic, similar to original rat movements thus restricting sudden change of directions

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
