
class Pig():
    # Constructor
    def __init__(self, T: int = 2):
        self.T = T
        self.S = [(i,j,k) for i in range(T) for j in range(T) for k in range(T-i)]
        self.S.reverse()
        self.A = {"roll","hold"}
        self.V = {s:0 for s in self.S}
        self.policy = {s: None for s in self.S}
        self.iter = 0
        self.converge = None

    # Define a winning state
    def isWin(self, s: tuple[int, int, int]) -> bool:
        return s[0] + s[2] >= self.T

    # Define a lossing state
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
    def value_action(self, s: tuple[int,int,int], a: str):
        if a == "roll":
            aux = sum([self.value((s[0],s[1],s[2]+r)) for r in range(2,7)])
            return (1.0 - self.value((s[1],s[0],0)) + aux) / 6
        elif a == "hold":
            return 1.0 - self.value((s[1],s[0]+s[2],0))

    # Value iteration algorithm
    def value_iteration(self, gamma: float = 1.0, tol: float = 1e-3, iter_max: int = 1000):
        if not (0 < gamma <= 1):  # Validate gamma
            raise ValueError(f"Value {gamma} is out of range (0, 1]")

        iteration_count = 1

        for iter in range(1,iter_max+1):
            delta = 0 
            for s in self.S:
                self.policy[s], aux_value = max(((a, self.value_action(s, a)) for a in self.A), key=lambda x: x[1])
                delta = max(delta, abs(aux_value - self.V[s]))   
                self.V[s] = aux_value  

            iteration_count += 1

            if delta < tol:
                self.iter = iter
                self.converge = True
                return
        
        self.iter = iter_max
        self.converge = False
        print(f"WARNING: Maximum number of iterations ({iter_max}) reached!")

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


