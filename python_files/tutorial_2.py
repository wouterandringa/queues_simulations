# %%
'''
# Tutorial 2, solutions


This solution is a jupyter notebook which allows you to directly interact with the code so that
you can see the effect of any changes you may like to make.

Author: Nicky van Foreest
'''

# %%
from heapq import heappop, heappush
import scipy
from scipy.stats import uniform, expon

scipy.random.seed(3)

def sort_ages():
    stack = []

    heappush(stack, (21, "Jan"))
    heappush(stack, (20, "Piet"))
    heappush(stack, (18, "Klara"))
    heappush(stack, (25, "Cynthia"))

    while stack:
        age, name = heappop(stack)
        print(name, age)

sort_ages()

# %%
def sort_ages_with_more_info():
    stack = []

    heappush(stack, (21, "Jan", "Huawei"))
    heappush(stack, (20, "Piet", "Apple"))
    heappush(stack, (18, "Klara", "Motorola"))
    heappush(stack, (25, "Cynthia", "Nexus"))

    while stack:
        age, name, phone = heappop(stack)
        print(age, name, phone)

sort_ages_with_more_info()

# %%
# void  code for numbering

# %%
ARRIVAL = 0
DEPARTURE = 1

stack = [] # this is the event stack

# %%
class Job:
    def __init__(self):
        self.arrival_time = 0
        self.service_time = 0
        self.departure_time = 0
        self.queue_length_at_arrival = 0

    def sojourn_time(self):
        return self.departure_time - self.arrival_time

    def waiting_time(self):
        return self.sojourn_time() - self.service_time

    def __repr__(self):
        return f"{self.arrival_time}, {self.service_time}, {self.departure_time}\n"

    def __le__(self, other):
        # this is necessary to sort jobs when they have the same arrival times.
        return self.id <= other.id


# %%
def experiment_1():
    labda = 2.0
    mu = 3.0
    rho = labda / mu
    F = expon(scale=1.0 / labda)  # interarrival time distributon
    G = expon(scale=1.0 / mu)  # service time distributon

    num_jobs = 10

    time = 0
    for i in range(num_jobs):
        time += F.rvs()
        job = Job()
        job.arrival_time = time
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        print(job)



experiment_1()

# %%
class Server:
    def __init__(self):
        self.busy = False

server = Server()
queue = []
served_jobs = [] # used for statistics

def start_service(time, job):
    server.busy = True
    job.departure_time = time + job.service_time
    heappush(stack, (job.departure_time, job, DEPARTURE))

def handle_arrival(time, job):
    job.queue_length_at_arrival = len(queue)
    if server.busy:
        queue.append(job)
    else:
        start_service(time, job)

def handle_departure(time, job):
    server.busy = False
    if queue: # queue is not empty
        next_job = queue.pop(0) # pop oldest job in queue
        start_service(time, next_job)

# %%
def experiment_2():
    labda = 2.0
    mu = 3.0
    rho = labda / mu
    F = expon(scale=1.0 / labda)  # interarrival time distributon
    G = expon(scale=1.0 / mu)  # service time distributon
    num_jobs = 10  # too small, change it to a larger number, and rerun the experiment

    time = 0
    for i in range(num_jobs):
        time += F.rvs()
        job = Job()
        job.arrival_time = time
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        if typ == ARRIVAL:
            handle_arrival(time, job)
        else:
            handle_departure(time, job)
            served_jobs.append(job)

    tot_queue = sum(j.queue_length_at_arrival for j in served_jobs)
    av_queue_length = tot_queue / len(served_jobs)
    print("Theoretical avg. queue length: ", rho * rho / (1 - rho))
    print("Simulated avg. queue length:", av_queue_length)

    tot_sojourn = sum(j.sojourn_time() for j in served_jobs)
    av_sojourn_time = tot_sojourn/len(served_jobs)
    print("Theoretical avg. sojourn time:", (1./labda)*rho/(1-rho))
    print("Simulated avg. sojourn time:", av_sojourn_time)

experiment_2()


# %%
def experiment_2a():
    labda = 2.0
    mu = 3.0
    rho = labda / mu
    F = expon(scale=1.0 / labda)  # interarrival time distributon
    G = expon(scale=1.0 / mu)  # service time distributon
    num_jobs = 1000

    time = 0
    for i in range(num_jobs):
        time += F.rvs()
        job = Job()
        job.arrival_time = time
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        if typ == ARRIVAL:
            handle_arrival(time, job)
        else:
            handle_departure(time, job)
            served_jobs.append(job)

    tot_queue = sum(j.queue_length_at_arrival for j in served_jobs)
    av_queue_length = tot_queue / len(served_jobs)
    print("Theoretical avg. queue length: ", rho * rho / (1 - rho))
    print("Simulated avg. queue length:", av_queue_length)

    tot_sojourn = sum(j.sojourn_time() for j in served_jobs)
    av_sojourn_time = tot_sojourn/len(served_jobs)
    print("Theoretical avg. sojourn time:", (1./labda)*rho/(1-rho))
    print("Simulated avg. sojourn time:", av_sojourn_time)

experiment_2a()

# %%
from scipy.stats import uniform

stack = [] # this is the event stack
queue = []
served_jobs = [] # used for statistics

def experiment_3():
    labda = 2.0
    mu = 3.0
    rho = labda / mu
    F = uniform(3, 0.00001)
    G = uniform(2, 0.00001)
    num_jobs = 10

    time = 0
    for i in range(num_jobs):
        time += F.rvs()
        job = Job()
        job.arrival_time = time
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        if typ == ARRIVAL:
            handle_arrival(time, job)
        else:
            handle_departure(time, job)
            served_jobs.append(job)

    for j in served_jobs:
        print(j)


experiment_3()

# %%

def pollaczek_khinchine(labda, G):
    ES = G.mean()
    rho = labda*ES
    ce2 = G.var()/ES/ES
    EW = (1.+ce2)/2 * rho/(1-rho)*ES
    return EW


labda = 1./3
F = expon(scale=1./labda)  # interarrival time distributon
G = uniform(1, 2)

print("PK: ", labda*pollaczek_khinchine(labda,G))

# %%
stack = [] # this is the event stack
queue = []
served_jobs = [] # used for statistics

def test_mg1():
    job = Job()
    labda = 1.0 / 3
    F = expon(scale=1.0 / labda)  # interarrival time distributon
    G = uniform(1, 3)
    print("ES: ", G.mean(), "rho: ", labda * G.mean())

    num_jobs = 1000

    time = 0
    for i in range(num_jobs):
        time += F.rvs()
        job = Job()
        job.arrival_time = time
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        if typ == ARRIVAL:
            handle_arrival(time, job)
        else:
            handle_departure(time, job)
            served_jobs.append(job)

    tot_queue = sum(j.queue_length_at_arrival for j in served_jobs)
    av_queue_length = tot_queue / len(served_jobs)
    print("Theoretical avg. queue length: ", labda * pollaczek_khinchine(labda, G))
    print("Simulated avg. queue length:", av_queue_length)

    tot_sojourn = sum(j.sojourn_time() for j in served_jobs)
    av_sojourn_time = tot_sojourn / len(served_jobs)
    print("Theoretical avg. sojourn time: ", pollaczek_khinchine(labda, G) + G.mean())
    print("Avg. sojourn time:", av_sojourn_time)

test_mg1()

# %%
import numpy as np

F = np.sort(uniform(0, 120).rvs(60))

# %%
from collections import Counter

stack = [] # this is the event stack
queue = []
served_jobs = [] # used for statistics

def check_in():
    num_jobs = 60
    F = np.sort(uniform(0, 120).rvs(num_jobs))
    G = uniform(1, 3)

    for i in range(num_jobs):
        job = Job()
        job.arrival_time = F[i]
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        if typ == ARRIVAL:
            handle_arrival(time, job)
        else:
            handle_departure(time, job)
            served_jobs.append(job)

    tot_queue = sum(j.queue_length_at_arrival for j in served_jobs)
    av_queue_length = tot_queue / len(served_jobs)
    print("Simulated avg. queue length:", av_queue_length)

    tot_sojourn = sum(j.sojourn_time() for j in served_jobs)
    av_sojourn_time = tot_sojourn / len(served_jobs)
    print("Avg. sojourn time:", av_sojourn_time)

    c = Counter([j.queue_length_at_arrival for j in served_jobs])
    print("Queue length distributon, sloppy output")
    print(sorted(c.items()))


check_in()

# %%
import matplotlib.pyplot as plt

x = [job.arrival_time for job in served_jobs]
y = [job.queue_length_at_arrival for job in served_jobs]

plt.plot(x, y, "o")
plt.show()

# %%
stack = [] # this is the event stack
queue = []
served_jobs = [] # used for statistics

def check_in_2():
    num_jobs = 60
    F = np.sort(uniform(0, 150).rvs(num_jobs))
    G = uniform(1, 2)

    for i in range(num_jobs):
        job = Job()
        job.arrival_time = F[i]
        job.service_time = G.rvs()
        heappush(stack, (job.arrival_time, job, ARRIVAL))

    while stack:
        time, job, typ = heappop(stack)
        if typ == ARRIVAL:
            handle_arrival(time, job)
        else:
            handle_departure(time, job)
            served_jobs.append(job)

    tot_queue = sum(j.queue_length_at_arrival for j in served_jobs)
    av_queue_length = tot_queue / len(served_jobs)
    print("Simulated avg. queue length:", av_queue_length)

    tot_sojourn = sum(j.sojourn_time() for j in served_jobs)
    av_sojourn_time = tot_sojourn / len(served_jobs)
    print("Avg. sojourn time:", av_sojourn_time)

    x = [job.arrival_time for job in served_jobs]
    y = [job.queue_length_at_arrival for job in served_jobs]
    
    plt.plot(x, y, "o")
    plt.show()
check_in_2()
