import warnings

class Pig():
    # Constructor
    def __init__(self, T: int = 2):
        self.T = T
        self.S = {(i,j,k) for i in range(T) for j in range(T) for k in range(T-i)}
        self.A = {"roll","hold"}
        self.V = {s:0 for s in self.S}
        self.policy = {s: None for s in self.S}
        self.iter = 0
        self.converge = None

    # Define a winning state
    def isWin(self, s: tuple[int, int, int]) -> bool:
        return s[0] + s[2] >= self.T

    # Terminal state (absorbing)
    def isLoss(self, s: tuple[int, int, int]) -> bool:
        return s[1] >= self.T

    # Value function constraint
    def value(self, s: tuple[int, int, int]):
        if self.isWin(s):
            return 1
        elif self.isLoss(s):
            return 0
        else:
            return self.V[s]

    # Value state-action
    def value_action(self,s: tuple[int,int,int], a: str):
        if a == "roll":
            aux = sum([self.value((s[0], s[1], s[2] + r)) for r in range(2,7)])
            return (1.0 - self.value((s[1], s[0], 0)) + aux) / 6
        elif a == "hold":
            return 1.0 - self.value((s[1],s[0],0))

    # Value iteration algorithm
    def value_iteration(self,gamma: float = 1, tol: float =1e-6, iter_max: iter =1000):
        if not (0 < gamma <= 1):  # Validate gamma
            raise ValueError(f"Value {gamma} is out of range (0, 1]")

        iteration_count = 1

        while True:
            delta = 0  # Track the maximum change in value function
            new_V = self.V.copy()
            for s in self.S:
                new_V[s] = max([self.value_action(s,a) for a in self.A])
                delta = max(delta, max(abs(self.V[r] - new_V[r]) for r in self.S))         
            self.V = new_V
            iteration_count += 1
            
            if delta < tol or iteration_count >= iter_max:
                if iteration_count >= iter_max:
                    self.converge = False
                    warnings.warn(f"Maximum number of iterations ({iter_max}) reached!", RuntimeWarning)
                break  # Stop if values converge
        
        self.iter = iteration_count
        self.converge = True

        # Extract the optimal policy
        for s in self.S:
            best_action = max(self.A, key=lambda a: self.value_action(s,a))
            self.policy[s] = best_action

    # Print policy method
    def print_policy(self):
        print("Optimal Policy:")
        for state, value in self.policy.items():
            print(f"{state}: {value}")

    # Print value function
    def print_value(self):
        print("Optimal Values:")
        for state, value in self.V.items():
            print(f"{state}: {value}")




