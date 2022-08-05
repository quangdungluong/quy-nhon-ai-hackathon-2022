from utils import get_final_prediction
import numpy as np

outputs = np.array([0, 0.5, 0.2, 0.3, 0.4, -0.1, -0.2, -0.2, -0.3, -0.4, 0, 0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.2, -0.3, -0.4, 0, 0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.2, -0.3, -0.4])

result = get_final_prediction(outputs=outputs)
print(result)
