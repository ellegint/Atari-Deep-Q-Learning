# Atari-Deep-Q-Learning
In this project I tried to create an AI that can play two different Atari games by training a single model\
For that purpose I used the Gym library to simulate the Atari environment and a Deep Q-Network to create the model.\
This was part of my university's Deep Learning course's final.

# The starting (wrong) idea
Given the task, I picked the two most similar games in Atari so that the model could be tricked to learn just one of the games and play well in both. In that sense, Breakout and Pong have been chosen: infact in these two games the player has to deal with a racket and a ball and in order not to lose they must not to make the ball fall on the bottom of the screen. The main difference is the reason and the way that the player has to hit the ball: in Pong they want to hit it so that the other player (the CPU) misses the ball, i.e. to score a point; in Breakout, instead, the ball has to be hit in a way such that it reaches a block that has not yet been destroyed.\
Of course one of those games had to be simulated rotated to be similar to the other one.

# The first (wrong) approach
That said, the first thing I tried to do was creating a model that could play well one of those games and then test its performances in both of them.
And that's what I did: I created the first model by training the network to play Pong and then test it on Breakout. You'll find the created model and the demos of this experiment inside the _Pong_ folder. The model can easily defeat the opponent playing Pong but it performs poorly on Breakout, as it could not even play properly. \
Then, I did the same with Breakout and obtained the same results (perfroms well on trained game, poorly on the other one). Also I found out that Breakout is much harder to learn for the model and for that reason I discovered the multienvironments, an instrument that could be used to create and play more simulations simultaneously.

# The final solution
The best way to make the model learn both of the games was (trivially) to make it play both of them. By using the multienvs I managed to do so and obtained some not-so-bad results, that can be found in the _Mixed_ folder.

# Repo Structure
Inside each folder you'll find the trained model for that approach and two videos showing how the model performs in the games.
Each of the _.py_ files has been used for the namesake approach (i.e. breakout.py has been used to train the breakout model that can be found in the folder of the same name) 

# How to use the code
Install the requirements (Pytorch is stictly necessary) and use the following functions in the _main_ of one of the scripts:
+ _train()_ to train a new model using the DQN. It could take several hours (we're talking days if you don't have a good GPU)
+ _test(game, model)_ to visually simulate a game of Pong or Breakout. That can be picked using the function parameters
+ _performance(game, model)_ to get just the results of the simulation, with no visualization
