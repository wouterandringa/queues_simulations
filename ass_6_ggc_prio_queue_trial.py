#My own trial of assignment 6: G/G/c with priority queueing


#IMPORTS
from collections import deque #used in 4.10
from heapq import heappop, heappush
import numpy as np
from scipy.stats import expon, uniform


#GLOBAL VARIABLES

#event types
ARRIVAL = 0
DEPARTURE = 1

#job types
PRIORITY = 3
NONPRIORITY = 4

#downserving (is it allowed for non-prio passengers to be served at prio desk?)
NOTALLOWED = False
ALLOWED = True



#CLASSES

#Job
class Job:
    def __init__(self, j_type):
        self.arrival_time = 0
        self.service_time = 0
        self.departure_time = 0
        self.queue_length_at_arrival = 0
        self.j_type = j_type
        self.s_type = 0   #type of server the job is served by
        
    def sojourn_time(self):
        return self.departure_time - self.arrival_time
    
    def waiting_time(self):
        return self.sojourn_time() - self.service_time
    
    def __repr__(self):
        return f"{self.arrival_time}, {self.service_time}, {self.departure_time}, {self.priority}\n"
    
#GGc_priority
class GGC_priority:
    def __init__(self, Fp, Fn, Gp, Gn, Cp, Cn, run_length, downserving=ALLOWED):
        self.Fp = Fp   #interarrival dist priority
        self.Fn = Fn   #interarrival dist non-priority
        self.Gp = Gp   #service dist priority
        self.Gn = Gn   #service dist non-priority
        self.Cp = Cp   #num servers priority
        self.Cn = Cn   #num servers non-priority
        self.Qp = deque()   #queue priority
        self.Qn = deque()   #queue non-priority
        self.run_length = run_length
        self.busy_p = 0   #num busy servers priority
        self.busy_n = 0   #num busy servers non-priority
        self.run_length = run_length
        self.stack = []   #event stack
        self.served_jobs = set()   #served jobs; for computing statistics
        self.downserving = downserving   #are non-priority passengers allowed to be helped by prio desk?
        
    def handle_arrival(self, time, job):
        self.generate_new_arrival(time, job.j_type)
        if job.j_type == PRIORITY:
            if self.busy_p < self.Cp:
                self.start_service(time, job, PRIORITY)   #serve at own server if possible
            elif self.busy_n < self.Cn:
                self.start_service(time, job, NONPRIORITY)   #serve at other server if possible
            else:
                self.Qp.append(job)   #put in its own queue
        else:
            if self.busy_n < self.Cn:
                self.start_service(time, job, NONPRIORITY)   #serve at own server if possible
            elif (self.busy_p < self.Cp and self.downserving):
                self.start_service(time, job, PRIORITY)   #serve at other server if possible and allowed
            else:
                self.Qn.append(job)   #put in its own queue

    def generate_new_arrival(self, time, j_type):
        if self.run_length > 0:
            self.run_length -= 1
            self.put_new_arrival_on_stack(time, j_type)
            
    def put_new_arrival_on_stack(self, time, j_type):
        job = Job(j_type)
        if job.j_type == PRIORITY:
            job.arrival_time = time + self.Fp.rvs()   #new arrival time
            job.service_time = self.Gp.rvs()   #new service time
        else:
            job.arrival_time = time + self.Fn.rvs()   #new arrival time
            job.service_time = self.Gn.rvs()   #new service time
        heappush(self.stack, (job.arrival_time, job, ARRIVAL))
        
    def start_service(self, time, job, s_type):
        job.departure_time = time + job.service_time
        job.s_type = s_type
        if s_type == PRIORITY:
            self.busy_p += 1
        else:
            self.busy_n += 1
        heappush(self.stack, (job.departure_time, job, DEPARTURE))
        
    def handle_departure(self, time, job):
        self.served_jobs.add(job)
        if job.s_type == PRIORITY:
            #consistency check:
            if(job.j_type == NONPRIORITY and self.downserving==False):
                print("Error: a non-prio was helped by a prio server, but this is not allowed")
                quit()
                
            self.busy_p -= 1
            if self.Qp:  #not empty
                next_job = self.Qp.popleft()
                self.start_service(time, next_job, PRIORITY)
            elif self.Qn and self.downserving:  #not empty and passenger is allowed to be served in prio queue
                next_job = self.Qn.popleft()
                self.start_service(time, next_job, PRIORITY)
        elif job.s_type == NONPRIORITY:
            self.busy_n -= 1
            if self.Qn:  #not empty
                next_job = self.Qn.popleft()
                self.start_service(time, next_job, NONPRIORITY)
            elif self.Qp:  #not empty
                next_job = self.Qp.popleft()
                self.start_service(time, next_job, NONPRIORITY)
        else:
            print("Error: we do not know the s_type of the job that is departing")
            quit()
             
    def consistency_check(self):
        if(self.busy_p < 0 or self.busy_p > self.Cp 
           or self.busy_n < 0 or self.busy_n > self.Cn or 
           len(self.Qp) < 0 or len(self.Qn) < 0 or 
           (len(self.Qp) > 0 and self.busy_p < self.Cp) or 
           (len(self.Qn) > 0 and self.busy_n < self.Cn)):
            
            print("There is something wrong")
            quit()
    
   
    def run(self):
        time = 0
        self.put_new_arrival_on_stack(time, PRIORITY)
        self.put_new_arrival_on_stack(time, NONPRIORITY)
        self.run_length -= 2
        
        while self.stack:  #not empty
            time, job, typ = heappop(self.stack)
            self.consistency_check()  #only when testing
            if typ == ARRIVAL:
                self.handle_arrival(time, job)
            else:
                self.handle_departure(time, job)
    

#TESTING
np.random.seed(5)

#Sakasegawa formula from previous assignment
def sakasegawa(F, G, c):
    labda = 1./F.mean()
    ES = G.mean()
    rho = labda*ES/c
    EWQ_1 = rho**(np.sqrt(2*(c+1)) - 1)/(c*(1-rho))*ES
    ca2 = F.var()*labda*labda
    ce2 = G.var()/ES/ES
    return (ca2 + ce2)/2 * EWQ_1

#Test an M/D/2 queue
def md2_test(labda=1.8, mu=1, num_jobs=100):
    F = expon(scale=1./(labda/2))  #split up arrivals between the two queues
    G = uniform(mu, 0.0001)
    
    ggc_prio = GGC_priority(F, F, G, G, 1, 1, num_jobs)
    ggc_prio.run()
    tot_wait_in_q = sum(j.waiting_time() for j in ggc_prio.served_jobs)
    avg_wait_in_q = tot_wait_in_q/len(ggc_prio.served_jobs)
        
    print("M/D/2 TEST")
    F_theory = expon(scale=1./(labda))  #merged arrivals
    print("Theoretical avg. waiting time in queue:", sakasegawa(F_theory, G, 2))
    print("Simulated avg. waiting time in queue:", avg_wait_in_q)

#md2_test(num_jobs=100)
#md2_test(num_jobs=10000)


#Test an M/M/8 queue
def mm8_test(labda=7, mu=1, num_jobs=100):
    F = expon(scale=1./(labda/2))  #split up arrivals between the two queues
    G = expon(scale = 1./mu)
    
    ggc_prio = GGC_priority(F, F, G, G, 4, 4, num_jobs)
    ggc_prio.run()
    tot_wait_in_q = sum(j.waiting_time() for j in ggc_prio.served_jobs)
    avg_wait_in_q = tot_wait_in_q/len(ggc_prio.served_jobs)
        
    print("M/M/8 TEST")
    F_theory = expon(scale=1./(labda))  #merged arrivals
    print("Theoretical avg. waiting time in queue:", sakasegawa(F_theory, G, 8))
    print("Simulated avg. waiting time in queue:", avg_wait_in_q)

#mm8_test(num_jobs=100)
#mm8_test(num_jobs=10000)



#EXPERIMENTS

#Think about some interesting things to analyze here

#Example: average waiting time in queue for both types of customers
def exper_avg_waiting_time(Fp, Fn, Gp, Gn, Cp, Cn, run_length, downserving=True):
    ggc_prio = GGC_priority(Fp, Fn, Gp, Gn, Cp, Cn, run_length, downserving)
    ggc_prio.run()
    tot_winQ_p = 0
    num_jobs_p = 0
    tot_winQ_n = 0
    num_jobs_n = 0
    for j in ggc_prio.served_jobs:
        if j.j_type == PRIORITY:
            num_jobs_p += 1
            tot_winQ_p += j.waiting_time()
        elif j.j_type == NONPRIORITY:
            num_jobs_n += 1
            tot_winQ_n += j.waiting_time()
    avg_winQ_p = tot_winQ_p/num_jobs_p
    avg_winQ_n = tot_winQ_n/num_jobs_n
    
    print("WAITING TIME EXPERIMENT")
    downserving_string = "NOT ALLOWED"
    if downserving:
        downserving_string = "ALLOWED"
    print("Downserving is {}".format(downserving_string))
    print("Avg. waiting time in queue for priority jobs:", avg_winQ_p)
    print("Avg. waiting time in queue for non-priority jobs:", avg_winQ_n)
    

#Experiment settings
labda_p = 0.5  #0.5 passengers per minute
labda_n = 3.5  #3.5 passengers per minute
Fp = expon(scale=1./labda_p)
Fn = expon(scale=1./labda_n)
Gp = uniform(0.5, 1)  #on average, 1 passenger per minute
Gn = uniform(0.5, 1)  #on average, 1 passenger per minute
Cp = 1
Cn = 4

run_length = 10000
    
#Run the experiment    
exper_avg_waiting_time(Fp, Fn, Gp, Gn, Cp, Cn, run_length, downserving=True)
exper_avg_waiting_time(Fp, Fn, Gp, Gn, Cp, Cn, run_length, downserving=False)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









