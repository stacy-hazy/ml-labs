import numpy as np # numerical python library for calculus
from matplotlib import pyplot as plt # library for creating static, animated, and interactive visualizations in Python
from sklearn import datasets # python library to implement machine learning models and statistical modelling
from sklearn.model_selection import train_test_split # for splitting the dataset

np.random.seed(1)

df = datasets.load_breast_cancer()

X, y = df.data, df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = X_train.astype('float128')
X_test = X_test.astype('float128')

n_samples, n_features = X_train.shape

def parameters_inititalization(m):
  """
  Ця функція ініціалізує вектор-рядок випадкових дійсних значень ваг форми (1, m),
  отриманих з нормального розподілу та зсув (довільне дійсне значення)

  Параметри:
  m -- кількість вхідних ознак для кожного навчального прикладу

  Повертає:
  W -- вектор-рядок ваг форми (1, m)
  b -- зсув (скаляр)
  """

  W = np.random.normal(0.0, 1, (1, m))
  b = 0.0

  return W, b

W, b = parameters_inititalization(n_features)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def forwardPropagate(X, W, b):
  """
  Ця функція обчислює лінійну комбінацію вхідних ознак та ваг, включаючи зсув і знаходить активаційне значення сигмоїди

  Параметри:
  X -- вхідна матриця форми (n_samples, n_features)
  W -- вектор-рядок ваг моделі форми (1, n_features)
  b -- зсув моделі (скаляр)

  Повертає:
  z -- загальна зважена сума вхідних ознак, включаючи зсув
  y_hat -- активаційне значення сигмоїди
  """

  z = np.dot(W, X.T) + b
  y_hat = sigmoid(z)

  return z, y_hat

z = forwardPropagate(X_train, W, b)
z, y_hat = forwardPropagate(X_train, W, b)


def cost(n, y_hat, y_true):
  """
  Ця функція обчислює усереднену втрату для задачі бінарної класифікації на всьому навчальному наборі даних

  Параметри:
  n -- загальна кількість навчальних прикладів
  y_hat -- активаційне значення сигмоїди (прогноз логістичної регресії)
  y_true -- істинний клас зображення (очікувана мітка прогнозу)

  Повертає:
  J --  усереднена втрата моделі для задачі бінарної класифікації на всьому навчальному наборі даних
  """
  ep = 10E-10 # для уникнення в log(0)

  J = (np.sum(y_true * np.log(y_hat + ep) + (1 - y_true) * np.log(1 - y_hat + ep))) / (-n)

  return J

J = cost(n_samples, y_hat, y_train)


def backwardPropagate(n, X, y_hat, y_true):
  """
  Ця функція обчислює градієнти цільової функції відносно ваг та зсуву

  Параметри:
  n -- загальна кількість навчальних прикладів
  X -- вхідна матриця форми (n_samples, n_features)
  y_hat -- активаційне значення сигмоїди (прогноз логістичної регресії)
  y_true -- істинний клас зображення (очікувана мітка прогнозу)

  Повертає:
  dW --  градієнт цільової функції відносно ваг моделі
  db -- градієнт цільової функції відносно зсуву моделі
  """

  dW = np.dot((y_hat - y_true), X) / n
  db = np.sum(y_hat - y_true) / n

  return dW, db

dW, db = backwardPropagate(n_samples, X_train, y_hat, y_train)


def update(lr, dW, db, W, b):
  """
  Ця функція оновлює навчальні параметри моделі (ваги та зсув) у напрямку мінімізації цільової функції

  Параметри:
  lr -- швидкість  навчання (крок навчання)
  dW --  градієнт цільової функції відносно ваг моделі
  db -- градієнт цільової функції відносно зсуву моделі
  W -- вектор-рядок ваг моделі форми (1, n_features)
  b -- зсув моделі (скаляр)

  Повертає:
  W -- оновлений вектор-рядок ваг моделі форми (1, n_features)
  b -- оновлений зсув моделі (скаляр)
  """

  W = W - lr * dW
  b = b - lr * db


W, b = update(0.0001, dW, db, W, b)



class LogisticRegression:

  def __init__(self, lr=0.001, n_iters=1000):
      self.lr = lr
      self.n_iters = n_iters


  def fit(self, X, y):
    """
    Trains a logistic regression model using gradient descent
    """
    # Step 0: Initialize the parameters
    n_samples, n_features = X.shape
    self.W, self.b = parameters_inititalization(n_features)

    costs = []

    for i in range(self.n_iters):
      # Step 1: Compute a linear combination of the input features and weights
      z, y_hat = forwardPropagate(X, self.W, self.b)
      # Step 2: Compute cost over training set
      J = cost(n_samples, y_hat, y)
      costs.append(J)
      if i % 20 == 0:
        print(f"Усереднена втрата моделі на ітерації {i}: {J}")
      # Step 3: Compute the gradients
      dW, db = backwardPropagate(n_samples, X, y_hat, y)
      # Step 4: Update the parameters
      self.W, self.b = update(self.lr, dW, db, self.W, self.b)
    return self.W, self.b, costs

  def predict(self, X):
    z, y_hat = forwardPropagate(X, self.W, self.b)
    class_pred = [0 if y < 0.5 else 1 for y in y_hat[0][:]]
    return class_pred

  def evaluate(self, X, y):
    n_samples, n_features = X.shape
    z, y_hat = forwardPropagate(X, self.W, self.b)
    J = cost(n_samples, y_hat, y)
    return J



learning_rates = [0.001, 0.01, 0.1, 0.5]
iterations = [100, 500, 1000, 5000]

results_lr = {}
for lr in learning_rates:
    model = LogisticRegression(lr=lr, n_iters=1000)
    _, _, costs = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_pred, y_test)
    results_lr[lr] = {'costs': costs, 'accuracy': acc}

# Візуалізація результатів для швидкості навчання
plt.figure(figsize=(8, 6))
for lr, data in results_lr.items():
    plt.plot(np.arange(1000), data['costs'], label=f"lr={lr}")
plt.title("Втрати на кожній ітерації для різних швидкостей навчання")
plt.xlabel("Ітерації")
plt.ylabel("Втрати")
plt.legend()
plt.show()

results_iters = {}
for iters in iterations:
    model = LogisticRegression(lr=0.01, n_iters=iters)
    _, _, costs = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_pred, y_test)
    results_iters[iters] = {'costs': costs, 'accuracy': acc}

# Візуалізація результатів для кількості ітерацій
plt.figure(figsize=(8, 6))
for iters, data in results_iters.items():
    plt.plot(np.arange(len(data['costs'])), data['costs'], label=f"iters={iters}")
plt.title("Втрати на кожній ітерації для різної кількості ітерацій")
plt.xlabel("Ітерації")
plt.ylabel("Втрати")
plt.legend()
plt.show()

# Виведення точності моделі для кожного експерименту
print("Результати експерименту зі швидкістю навчання:")
for lr, data in results_lr.items():
    print(f"Швидкість навчання {lr}: точність {data['accuracy']:.4f}")

print("\nРезультати експерименту з кількістю ітерацій:")
for iters, data in results_iters.items():
    print(f"Кількість ітерацій {iters}: точність {data['accuracy']:.4f}")
