import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.terminated = torch.zeros((max_size, 1))
        self.truncated = torch.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: float,
        terminated: float,
        truncated: float,
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        # self.not_done[self.ptr] = 1.0 - done
        self.terminated[self.ptr] = terminated
        self.truncated[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.terminated[ind]).to(self.device),
            torch.FloatTensor(self.truncated[ind]).to(self.device),
        )
