import numpy as np 

def FluidVelocity(density, massflowrate, Ac):
    return massflowrate / ( density * Ac )


#D is tube diameter 
#Velocity is mean velocity
#mu is viscosity 
def Red(density, velocity, D, mu):
    return ( density * velocity * D ) / mu

#Red = 2300 is onset of turbulence 
#Red = 10000 is when it starts to get fully turbulent 


#For fully developed (hydrodynamically and thermally) turbulent flow in a smooth circular tube, 
# the local Nusselt number may be obtained from the Dittus Boelter equation


#pg 409
#mu is viscosity
#ThermalConductivity    
def Pr(Cp, mu, k):
    return ( Cp * mu ) / k


def f(Red):
    if Red < 2300:
        return 64 / Red
    
    if Red >= 3000:
        return (0.790 * np.log(Red) - 1.64)**(-2)

    else: 
        return 'Red in transient state'




def Nud(Red, Pr, n, f):
    #TURBULEUNT
    n = 0.3
    if 10000 <= Red <= 30000: 
        #These conditions are for only: 
        #0.6 <= Pr <= 160
        #Red > 10,000
        #L/D > 10
        return 0.023 * Red**(4/5) * Pr**n

    if Red > 30000:
        #More Complex analysis 
        #Need f from moody chart
        #Evaluate Properties at Tm
        #0.5 < Pr < 2000
        #3000 < Red < 5e6
        return ( (f/8) * ( Red - 1000) * Pr ) / ( 1 + 12.7 * ( f / 8 )**n * ( (Pr**(2/3)) - 1 ) ) 
    
    if Red < 2300:
        return 3.66


def h(k, D, Nud):
    return ( k / D ) * Nud


