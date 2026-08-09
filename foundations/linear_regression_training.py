import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X) 
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        ## Parameters of the the model --> y = m1x+c1



        # model_predictions = self.get_model_predictions(X,initial_weights)
        # for i in range(3): 
        #     derivative = self.get_derevative(model_predictions, Y, len(X),X, initial_weights)



        der = np.zeros(3)
        weight = initial_weights
        for i in range(num_iterations):

            model_predictions = self.get_model_prediction(X,weight)
            for j in range(3): ## This will be used to get the derivative and the update the corresponding weights: 

                der[j] = self.get_derivative(model_predictions, Y, len(X),X,j)
                weight[j] = weight[j] - 0.01*der[j]
        
        return np.round(weight,5)
