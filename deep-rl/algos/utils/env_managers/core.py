import torch 
import numpy as np


class EnvManager():
    def __init__(self, env, device):
        self.device = device 
        self.env = env.unwrapped # unwrapped gives us access to the behind the scenes dynamics of the environment
        self.env.reset()
        self.current_screen = None # track the screen (render) of the environment. When None, we are at the beginning of the episode 
        self.done = False # episode's state
     

    def crop_screen(self, s):
        raise ("raise NotImplementedError")
        
    def transform_screen_data(self, s):
        raise ("raise NotImplementedError")
    
    def reset(self):
        self.env.reset()
        self.current_screen = None
    
    def close(self):
        self.env.close()
    
    def render(self, mode = "human"):
        return self.env.render(mode) # render the current state to the screen
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):
        next_obs, reward, self.done, info = self.env.step(action.item())
        return torch.tensor([reward], device = self.device)
    
    def just_starting(self):
        return self.current_screen is None # if True starting state
    
    def get_state(self):
        if self.just_starting() or self.done: # starting or ending of an episode
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
        
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_processed_screen(self):
        screen = self.render("rgb_array").transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    
    