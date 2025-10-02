
import numpy as np
import pickle


#---- A->C->E ----#
def train_baseline(df):
    y = df["travel_time"].to_numpy(dtype=float)
    avg_time = float(np.mean(y))
    yhat = np.full_like(y, avg_time, dtype=float)

    loss = loss_mse(yhat, y)
    return {"type": "baseline", "avg_time":avg_time, "loss": loss}

def predict_baseline(model, t_min):
    return  model["avg_time"]


#---- A->C->D ----#
def train_quadratic(df, n_iter=10000):

    x = df["depature"].to_numpy(dtype=float)
    y = df["travel_time"].to_numpy(dtype=float)

    best_loss = float("inf")
    best_theta = None

    for i in range (n_iter):

        a = np.random.uniform(-0.0001, 0.0001)
        b = np.random.uniform(-1, 1)
        c = np.random.uniform(y.min(), y.max())
        theta = (a, b, c)

        yhat = quadratic_fn(x, theta)
        loss = loss_mse(yhat, y)

        if loss < best_loss:
            best_loss = loss
            best_theta = theta

    return {"type":"quadratic", "theta": best_theta, "loss": best_loss}

def quadratic_fn(x, theta):
    a, b, c = theta
    return  a * x**2 + b*x + c

def predict_quadratic(model, t):
    return float(quadratic_fn(t, model["theta"]))


#---- B->C->E ----#
def train_sinus(df, n_iter=3000, seed=None):

    x = df["depature"].to_numpy(dtype=float)
    y = df["travel_time"].to_numpy(dtype=float)

    avg_time = float(y.mean())
    sigma = float(y.std()) or 1.0

    #  picks a frequency
    omega = np.random.uniform(2 * np.pi/600, 2*np.pi/200)

    best_loss, best_theta = float("inf"), None

    # Tries many different combinations of amplitude, phase and the average
    for i in range(n_iter):
        A = np.random.uniform(0, 3 * sigma)
        phi = np.random.uniform(0, 2 * np.pi)
        baseline = np.random.uniform(avg_time - 2*sigma, avg_time + 2*sigma)


        theta = (A, omega, phi, baseline)
        yhat = sinus_fn(x, theta)
        loss = loss_mse(yhat, y)

        if loss < best_loss:
            best_loss, best_theta  = loss, theta

    return {"type": "sinus", "theta": best_theta, "loss": best_loss}

def sinus_fn(x, params):
    A, omega, phi, baseline = params
    return A * np.sin(omega * x + phi) + baseline

def predict_sinus(model, t_min):
    return float(sinus_fn(float(t_min), model["theta"]))


#--- B->C->D ----#

def train_sinus2_fn(df, n_iter=3000):

    x = df["depature"].to_numpy(dtype=float)
    y = df["travel_time"].to_numpy(dtype=float)

    avg_time = float(y.mean())
    sigma = float(y.std()) or 1.0

    omega_lo, omega_hi = 2 * np.pi/600, 2 * np.pi/200

    best_loss, best_theta = float("inf"), None

    for i in range(n_iter):
        omega = np.random.uniform(omega_lo, omega_hi)
        A1 = np.random.uniform(0, 3 * sigma)
        A2 = np.random.uniform(0, 2 * sigma)

        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = np.random.uniform(0, 2 * np.pi)

        baseline = np.random.uniform(avg_time - 2*sigma, avg_time + 2*sigma)

        theta = (A1, A2, omega, phi1, phi2, baseline)

        yhat = sinus2_fn(x, theta)
        loss = loss_mse(yhat, y)

        if loss < best_loss:
            best_loss, best_theta  = loss, theta

    return {"type": "sinus2", "theta": best_theta, "loss": best_loss}


def sinus2_fn(x, theta):
    A1, A2, omega, phi1, phi2, baseline = theta
    return (baseline + A1 * np.sin(omega * x + phi1) + A2 * np.sin(2 * omega * x + phi2))


def predict_sinus2(model, t_min):
    return float(sinus2_fn(float(t_min), model["theta"]))


def loss_mse(yhat, y):
    d = yhat - y
    return np.mean(d ** 2)



def save_models(models, filename='trained_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {filename}")

def load_models(filename='trained_model.pkl'):
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    return models


def predict_generic(model, t_min: int) -> float:

    t = float(t_min)
    typ = model["type"]
    if typ == "baseline":
        return predict_baseline(model, t)
    if typ == "quadratic":
        return predict_quadratic(model, t)
    if typ == "sinus":
        return predict_sinus(model, t)
    if typ == "sinus2":
        return predict_sinus2(model, t)
    raise ValueError(f"Unoown model type: {model['type']}")


def find_best_route(dep_hour: int, dep_min: int, models: dict):

    t = int(dep_hour) * 60 + int(dep_min)
    preds = {road: predict_generic(m, t) for road, m in models.items()}
    best = min(preds, key=preds.get)
    return best, round(float(preds[best]), 1)



