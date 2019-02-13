from collections import deque
from heapq import heappop, heappush
import numpy as np
from scipy.stats import expon, uniform

np.random.seed(8)

ARRIVAL = 0
DEPARTURE = 1


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


class GGc:
    def __init__(self, F, G, c, run_length):
        self.F = F
        self.G = G
        self.c = c
        self.num_busy = 0  # number of busy servers
        self.run_length = run_length
        self.stack = []  # event stack
        self.queue = deque()
        self.served_jobs = set()

    def handle_arrival(self, time, job):
        self.generate_new_arrival(time)
        job.queue_length_at_arrival = len(self.queue)
        if self.num_busy < self.c:
            self.start_service(time, job)
        else:
            self.queue.append(job)

    def generate_new_arrival(self, time):
        if self.run_length > 0:
            self.put_new_arrival_on_stack(time)
            self.run_length -= 1

    def put_new_arrival_on_stack(self, time):
        job = Job()
        job.arrival_time = time + self.F.rvs()  # new arrival time
        job.service_time = self.G.rvs()  # new service time
        heappush(self.stack, (job.arrival_time, job, ARRIVAL))

    def start_service(self, time, job):
        self.num_busy += 1  # server becomes busy.
        job.departure_time = time + job.service_time
        heappush(self.stack, (job.departure_time, job, DEPARTURE))

    def handle_departure(self, time, job):
        self.num_busy -= 1
        self.served_jobs.add(job)
        if self.queue:  # not empty
            next_job = self.queue.popleft()
            self.start_service(time, next_job)

    def consistency_check(self):
        if (
            self.num_busy < 0
            or self.num_busy > self.c
            or len(self.queue) < 0
            or (len(self.queue) > 0 and self.num_busy < self.c)
        ):
            print("there is something wrong")
            quit()

    def run(self):
        time = 0
        self.put_new_arrival_on_stack(time)

        while self.stack:  # not empty
            time, job, typ = heappop(self.stack)
            # self.consistency_check() # use only when testing.
            if typ == ARRIVAL:
                self.handle_arrival(time, job)
            else:
                self.handle_departure(time, job)


def sakasegawa(F, G, c):
    labda = 1.0 / F.mean()
    ES = G.mean()
    rho = labda * ES / c
    EWQ_1 = rho ** (np.sqrt(2 * (c + 1)) - 1) / (c * (1 - rho)) * ES
    ca2 = F.var() * labda * labda
    ce2 = G.var() / ES / ES
    return (ca2 + ce2) / 2 * EWQ_1


def mm1_test(labda=0.8, mu=1, num_jobs=100):
    c = 1
    F = expon(scale=1.0 / labda)
    G = expon(scale=1.0 / mu)

    ggc = GGc(F, G, c, num_jobs)
    ggc.run()
    tot_wait_in_q = sum(j.waiting_time() for j in ggc.served_jobs)
    avg_wait_in_q = tot_wait_in_q / len(ggc.served_jobs)

    print("M/M/1 TEST")
    print("Theo avg. waiting time in queue:", sakasegawa(F, G, c))
    print("Simu avg. waiting time in queue:", avg_wait_in_q)


# mm1_test(num_jobs=100)
# mm1_test(num_jobs=100_000)


def md1_test(labda=0.9, mu=1, num_jobs=100):
    c = 1
    F = expon(scale=1.0 / labda)
    G = uniform(mu, 0.0001)

    ggc = GGc(F, G, c, num_jobs)
    ggc.run()
    tot_wait_in_q = sum(j.waiting_time() for j in ggc.served_jobs)
    avg_wait_in_q = tot_wait_in_q / len(ggc.served_jobs)

    print("M/D/1 TEST")
    print("Theo avg. waiting time in queue:", sakasegawa(F, G, c))
    print("Simu avg. waiting time in queue:", avg_wait_in_q)


# md1_test(num_jobs=100)
# md1_test(num_jobs=100_000)


def md2_test(labda=1.8, mu=1, num_jobs=100):
    c = 2
    F = expon(scale=1.0 / labda)
    G = uniform(mu, 0.0001)

    ggc = GGc(F, G, c, num_jobs)
    ggc.run()
    tot_wait_in_q = sum(j.waiting_time() for j in ggc.served_jobs)
    avg_wait_in_q = tot_wait_in_q / len(ggc.served_jobs)

    print("M/D/2 TEST")
    print("Theo avg. waiting time in queue:", sakasegawa(F, G, c))
    print("Simu avg. waiting time in queue:", avg_wait_in_q)


# md2_test(num_jobs=100)
# md2_test(num_jobs=10_000)


def intake_process(F, G, c, num):

    ggc = GGc(F, G, c, num)
    ggc.run()

    max_waiting_time = max(j.waiting_time() for j in ggc.served_jobs)
    longer_ten = sum((j.waiting_time() >= 10) for j in ggc.served_jobs)
    return max_waiting_time, longer_ten


def intake_test_1():
    labda = 300 / 60
    F = expon(scale=1.0 / labda)
    G = uniform(1, 2)
    num = 300

    print("Num servers, max waiting time, num longer than 10")
    for c in range(3, 10):
        max_waiting_time, longer_ten = intake_process(F, G, c, num)
        print(c, max_waiting_time, longer_ten)



# intake_test_1()


def intake_test_2():
    labda = 300 / 90
    F = expon(scale=1.0 / labda)
    G = uniform(1, 2)
    num = 300

    print("Num servers, max waiting time, num longer than 10")
    for c in range(3, 10):
        max_waiting_time, longer_ten = intake_process(F, G, c, num)
        print(c, max_waiting_time, longer_ten)



#intake_test_2()
