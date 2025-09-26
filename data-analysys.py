import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_traffic_data():
    """Load and parse traffic data from JSON file"""
    routes = defaultdict(lambda: {'times': [], 'durations': []})

    with open('traffic.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            route = data['road']

            # Parse times (note the typo in original data)
            dep_hour, dep_min = map(int, data['depature'].split(':'))
            arr_hour, arr_min = map(int, data['arrival'].split(':'))

            # Convert to minutes since midnight
            dep_time = dep_hour * 60 + dep_min
            arr_time = arr_hour * 60 + arr_min
            travel_time = arr_time - dep_time

            routes[route]['times'].append(dep_time)
            routes[route]['durations'].append(travel_time)

    return routes


def minutes_to_time_str(minutes):
    """Convert minutes since midnight to HH:MM format"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def plot_traffic_data():
    """Create comprehensive plots of the traffic data"""
    routes = load_traffic_data()

    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Traffic Data Analysis from traffic.jsonl', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange']
    route_names = list(routes.keys())

    # Plot 1: Scatter plot of all routes
    ax1 = axes[0, 0]
    for i, route in enumerate(route_names):
        times = np.array(routes[route]['times'])
        durations = np.array(routes[route]['durations'])
        ax1.scatter(times, durations, alpha=0.6, label=route, color=colors[i], s=20)

    ax1.set_xlabel('Departure Time (minutes since midnight)')
    ax1.set_ylabel('Travel Duration (minutes)')
    ax1.set_title('Travel Duration vs Departure Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add time labels on x-axis
    time_ticks = [360, 480, 600, 720, 840, 960, 1080, 1200]  # 6AM, 8AM, 10AM, etc.
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels([minutes_to_time_str(t) for t in time_ticks], rotation=45)

    # Plot 2: Distribution of travel times by route
    ax2 = axes[0, 1]
    all_durations = [routes[route]['durations'] for route in route_names]
    ax2.boxplot(all_durations, labels=route_names)
    ax2.set_ylabel('Travel Duration (minutes)')
    ax2.set_title('Distribution of Travel Times by Route')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average travel time by hour of day
    ax3 = axes[1, 0]
    for i, route in enumerate(route_names):
        times = np.array(routes[route]['times'])
        durations = np.array(routes[route]['durations'])

        # Group by hour
        hourly_avg = defaultdict(list)
        for t, d in zip(times, durations):
            hour = t // 60
            hourly_avg[hour].append(d)

        # Calculate averages
        hours = sorted(hourly_avg.keys())
        avg_durations = [np.mean(hourly_avg[h]) for h in hours]

        ax3.plot(hours, avg_durations, marker='o', label=route, color=colors[i], linewidth=2)

    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Average Travel Duration (minutes)')
    ax3.set_title('Average Travel Time by Hour')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(6, 20, 2))  # 6 AM to 6 PM, every 2 hours

    # Plot 4: Number of trips by route and hour
    ax4 = axes[1, 1]
    route_counts = {}
    for route in route_names:
        times = np.array(routes[route]['times'])
        hourly_counts = defaultdict(int)
        for t in times:
            hour = t // 60
            hourly_counts[hour] += 1

        hours = sorted(hourly_counts.keys())
        counts = [hourly_counts[h] for h in hours]
        route_counts[route] = (hours, counts)

    # Stacked bar chart
    bottom = np.zeros(len(range(6, 19)))  # 6 AM to 6 PM
    hours_range = list(range(6, 19))

    for i, route in enumerate(route_names):
        hours, counts = route_counts[route]
        # Align counts with hours_range
        aligned_counts = []
        for h in hours_range:
            if h in hours:
                idx = hours.index(h)
                aligned_counts.append(counts[idx])
            else:
                aligned_counts.append(0)

        ax4.bar(hours_range, aligned_counts, bottom=bottom, label=route, color=colors[i], alpha=0.8)
        bottom += np.array(aligned_counts)

    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Number of Trips')
    ax4.set_title('Trip Distribution by Hour')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('traffic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_data_summary():
    """Print summary statistics of the traffic data"""
    routes = load_traffic_data()

    print("=" * 60)
    print("TRAFFIC DATA SUMMARY")
    print("=" * 60)

    total_trips = 0
    for route in routes:
        durations = np.array(routes[route]['durations'])
        times = np.array(routes[route]['times'])

        print(f"\nRoute: {route}")
        print(f"  Number of trips: {len(durations)}")
        print(f"  Average duration: {durations.mean():.1f} minutes")
        print(f"  Min duration: {durations.min():.1f} minutes")
        print(f"  Max duration: {durations.max():.1f} minutes")
        print(f"  Std deviation: {durations.std():.1f} minutes")
        print(f"  Time range: {minutes_to_time_str(times.min())} to {minutes_to_time_str(times.max())}")

        total_trips += len(durations)

    print(f"\nTotal trips in dataset: {total_trips}")
    print("=" * 60)


if __name__ == '__main__':
    print("Loading and analyzing traffic data...")
    print_data_summary()
    print("\nGenerating plots...")
    plot_traffic_data()
    print("Analysis complete! Check 'traffic_analysis.png' for the visualization.")