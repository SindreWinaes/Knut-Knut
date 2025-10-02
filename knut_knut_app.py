import random
from flask import Flask
from flask import request
from train import load_models, find_best_route

app = Flask(__name__)

models = load_models()

def get_the_best_route_as_a_text_informatic(dep_hour, dep_min):
    best_route, estimated_time = find_best_route(int(dep_hour), int(dep_min), models)

    out = """
    <p>
    Departure time: {}:{} <br> 
    Best travel route: {} <br> 
    Estimated travel time of {} minutes. </p> 
    <p><a href="/">Back</a></p>
    """.format(dep_hour, dep_min, best_route, estimated_time)

    return out


@app.route('/')
def get_departure_time():
    return """
    	<h3>Knut Knut Transport AS</h3>
        <form action="/get_best_route" method="get">
            <label for="hour">Hour:</label>
            <select name="hour" id="hour">
                <option value="06">06</option>
                <option value="07">07</option>
                <option value="08">08</option>
                <option value="09">09</option>
                <option value="10">10</option>
                <option value="11">11</option>
                <option value="12">12</option>
                <option value="13">13</option>
                <option value="14">14</option>
                <option value="15">15</option>
                <option value="16">16</option>     
            </select>
            
            <label for="mins">Mins:</label>
            <input type="text" name="mins" size="2"/>
            <input type="submit">
        </form>
    """


@app.route("/get_best_route")
def get_route():
    departure_h = request.args.get('hour')
    departure_m = request.args.get('mins')

    route_info = get_the_best_route_as_a_text_informatic(departure_h, departure_m)
    return route_info


if __name__ == '__main__':
    print("<starting>")
    app.run()
    print("<done>")

