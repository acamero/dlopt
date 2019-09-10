import os
import numpy as np

#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

# The "black-box" objective function
def obj_func(x):
   x_r, x_i, x_d = np.array([x['C_0'],x['C_1']]), x['I'], x['N']
   if x_d == 'OK':
       tmp = 0
   else:
       tmp = 1
   print(x)
   return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.


# First we need to define the Search Space
# the search space consists of two continues variable
# one ordinal (integer) variable
# and one categorical.
C = ContinuousSpace([-5, 5],'C') * 2 
#here we defined two variables at once using the same lower and upper bounds.
#One with label C_0, and the other with label C_1
I = OrdinalSpace([-100, 100],'I') # one integer variable with label I
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], 'N')

#the search space is simply the product of the above variables
search_space = C * I * N

#next we define the surrogate model and the optimizer.
model = RandomForest(levels=search_space.levels)
opt = mipego(search_space, obj_func, model, 
                 minimize=True,     #the problem is a minimization problem.
                 max_eval=15,      #we evaluate maximum 500 times
                 max_iter=15,      #we have max 500 iterations
                 infill='EI',       #Expected improvement as criteria
                 n_init_sample=10,  #We start with 10 initial samples
                 n_point=1,         #We evaluate every iteration 1 time
                 n_job=1,           #  with 1 process (job).
                 optimizer='MIES',  #We use the MIES internal optimizer.
                 verbose=True, random_seed=None)


#and we run the optimization.
incumbent, stop_dict = opt.run()
