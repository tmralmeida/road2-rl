# Deep Reinforcement Learning Algorithms

## Code Template

I opted to use a transversal "code template" for deep RL algorithms based on [deeplizard tutorials](#https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv) because it is cleaner, reproducible and suitable for debugging. It is mainly composed of several different modules:

* Policy (and target) network(s) and/or state/state-action value functions 
* Experience is a *namedtuple* - e.g. in DQN is $(s,a,s',r)$
*  Replay memory for the algorithms, which make use of it, with the following functions:
  * appending new experiences
  * sampling experiences
  * other functions that can be useful for the respective algorithm
* An object that keeps track of the exploration/exploitation trade-off if off-policy method
* RL Agent  defines the actions of the agent or Actor Critic objects 
* Environment Manager is the high-level of the each environment from [Open AI Gym](#https://gym.openai.com/) 
* Utility functions (plot, metric, etc.)
* Main loop with the training 





## Policy gradient methods

* Estimate of the gradient of the policy's performance w.r.t. the respective policy parameters
* **Score Function Estimator** is a general method for estimating gradients of expectations: $\nabla_\theta E_x[f(x)] = E_x [\nabla_\theta \log p(x|\theta) f(x)]$. In Deep RL setting, let's say that $x$ is a random variable (the state or a trajectory) and $f(x)$ the function that maps the state\trajectory to a reward, then: $\nabla_\theta E_\tau[R(\tau)] = E_\tau [\nabla_\theta \log p(\tau|\theta) R(\tau)]$, which with a little manipulation (chain rule in the trajectory) is the same as:  $\nabla_\theta E_\tau[R(\tau)] = E_\tau [\sum^{T-1}_{t=0}\nabla_\theta \log \pi(a_t|s_t, \theta) R(\tau)]$
* If the reward is high, we want to move the parameters in order to increase the log likelihood of that trajectory
* We can reduce the variance of the policy gradient estimator by using a baseline; a near optima choice is the state-value function



### Vanilla Policy Gradient or REINFORCE

* Key idea of Policy Gradients algorithms:  estimate the gradient of the policy's performance, pushing the probabilities of actions that lead to higher return, and pushing down the probabilities of actions that lead to lower return, until arrive to the optimal policy
* On-Policy method, i.e. behavior policy (the one used to generate behavior) is the same as the target policy (the one that we want to find)
* Target environments: discrete/continuous
* Instead of state/state-action value functions, this algorithm make use of the advantage (*A*) of a policy over the others;  in practice, it is computed based on the infinite-horizon discounted return, despite the theoretical formulation: $\nabla_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta} [\sum^{T}_{t=0} \nabla_\theta \log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t, a_t)]$, where the advantage is given by $||b(s_t) - R_t||^2$, being $b$ the baseline 

* Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima
* Spinning up makes use of Generalized Advantage Estimation for computing the policy gradient
* [Pseudo-code](#https://spinningup.openai.com/en/latest/algorithms/vpg.html#vanilla-policy-gradient) 

