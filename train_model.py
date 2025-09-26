import json
import numpy as np
import pickle

ROUTES = ["A->C->D", "A->C->E","B->C->D", "B->C->E"]

def load_data():

    times = {}
    durs = {}

    for r in ROUTES:
        times[r] = []
        durs[r] = []

    with open("traffic.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            route = rec.get("road")
            dep = rec.get("depature")
            arr = rec.get("arrival")

            dep_time =  to_minutes(*map(int, dep.split(":")))
            arr_time = to_minutes(*map(int, arr.split(":")))

            travel_time = arr_time - dep_time

            times[route].append(dep_time)
            durs[route].append(travel_time)

    return times, durs


# Converts the time to a single number
def to_minutes(h,m):
    return int(h) * 60 + int(m)

def predict_h(t, theta):
    baseline, amplitude, frequency = theta
    return baseline + amplitude * np.sin(frequency * t)


def sample_theta (n=3, low=0, high=200, size=1):
    return np.random.uniform(low, high, size=(size, n))

def loss_mse(yhat, y):
    d = yhat - y
    return np.mean(d ** 2)

# Fortuna batch random search for one route
def train_route(times, durs, total=200000):

    best_theta = sample_theta()[0]
    best_loss = float("inf")

    print(f" Running {total} iterations...")

    for i in range(total):

        th = sample_theta()[0]
        curr_loss = loss_mse(predict_h(times, th), durs)

        if curr_loss < best_loss:
            best_loss, best_theta = curr_loss, th

    return best_theta, best_loss


def predict_travel_time(route, departure_hour, departure_min, models):
    if route not in models:
        return None

    departure_time = departure_hour * 60 + departure_min
    theta = models[route]['theta']
    predicted_time = predict_h(np.array([departure_time]), theta)[0]

    return predicted_time


def find_best_route(departure_hour, departure_min, models):
    best_route = None
    best_time = float('inf')

    for route in models:
        time = predict_travel_time(route, departure_hour, departure_min, models)
        if time and time < best_time:
            best_time = time
            best_route = route

    return best_route, best_time

def save_models(models, filename='trained_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {filename}")

def load_models(filename='trained_model.pkl'):
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    return models

def main():

    times, durs = load_data()
    models = {}

    for route in ROUTES:
        print(f"Fitting model for {route}...")

        t = np.array(times[route])
        y = np.array(durs[route])

        theta, loss = train_route(t, y)

        models[route] = {
            'theta':theta,
            'loss':loss
        }

        print(f"best loss = {loss}")
        print(f"best theta = {theta}")

    save_models(models)
    return models



if __name__ == '__main__':
    main()