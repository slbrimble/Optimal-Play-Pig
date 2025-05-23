
class Piglet():
    # Constructor
    def __init__(self, T: int = 2):
        self.T = T
        self.S = [(i,j,k) for i in range(T) for j in range(T) for k in range(T-i)]
        self.S.reverse()
        self.A = {"flip","hold"}
        self.V = {s:0 for s in self.S}
        self.policy = {s: None for s in self.S}
        self.trace = {s:[0] for s in self.S}
        self.iter = 0
        self.converge = None

    # Define a winning state
    def isWin(self, s: tuple[int, int, int]) -> bool:
        return s[0] + s[2] >= self.T

    # Terminal state (absorbing)
    def isLoss(self, s: tuple[int, int, int]) -> bool:
        return s[1] >= self.T

    # Value function constrained
    def value(self, s: tuple[int, int, int]):
        if self.isWin(s):
            return 1
        elif self.isLoss(s):
            return 0
        else:
            return self.V[s]

    # Value function state-action
    def value_action(self,s: tuple[int,int,int], a: str):
        if a == "flip":
            return (1.0 - self.value((s[1],s[0],0)) + self.value((s[0], s[1], s[2] + 1))) / 2
        elif a == "hold":
            return 1.0 - self.value((s[1],s[0]+s[2],0))

    # Value iteration
    def value_iteration(self,gamma: float = 1, tol: float =1e-6, iter_max: int = 1000):
        if not (0 < gamma <= 1):  # Validate gamma
            raise ValueError(f"Value {gamma} is out of range (0, 1]")

        iteration_count = 1

        while True:
            delta = 0  # Track the maximum change in value function
            new_V = self.V.copy()
            for s in self.S:
                self.policy[s], new_V[s] = max(((a, self.value_action(s, a)) for a in self.A), key=lambda x: x[1])
                self.trace[s].append(new_V[s])
                delta = max(delta, max(abs(self.V[r] - new_V[r]) for r in self.S))         
            self.V = new_V
            iteration_count += 1
            
            if delta < tol or iteration_count >= iter_max:
                if iteration_count >= iter_max:
                    self.iter = iter_max
                    self.converge = False
                    print(f"WARNING: Maximum number of iterations ({iter_max}) reached!")
                break
        
        if self.converge is None:
            self.iter = iteration_count
            self.converge = True

    # Print policy
    def print_policy(self):
        print("Optimal Policy:")
        for state, value in self.policy.items():
            print(f"{state}: {value}")

    # Print value function
    def print_value(self):
        print("Optimal Values:")
        for state, value in self.V.items():
            print(f"{state}: {value}")




