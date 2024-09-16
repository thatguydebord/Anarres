import numpy as np
import CoolProp as CP
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj


import Input as IP
import Thermal_Model as TM
import Flow_Equations as FE

import matplotlib.pyplot as plt


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


#   ░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░▒▓███████▓▒░       ░▒▓███████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓██████████████▓▒░  
#   ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
#   ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
#   ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓███████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒▒▓███▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
#   ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
#   ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
#   ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 

                                                                                                                                                               
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 




#------------------------------------------------------------------------------------------------------------------------
#  _____ _   _ _____ _______ _____          _         _____ ____  _   _ _____ _____ _______ _____ ____  _   _  _____ 
# |_   _| \ | |_   _|__   __|_   _|   /\   | |       / ____/ __ \| \ | |  __ |_   _|__   __|_   _/ __ \| \ | |/ ____|
#   | | |  \| | | |    | |    | |    /  \  | |      | |   | |  | |  \| | |  | || |    | |    | || |  | |  \| | (___  
#   | | | . ` | | |    | |    | |   / /\ \ | |      | |   | |  | | . ` | |  | || |    | |    | || |  | | . ` |\___ \ 
#  _| |_| |\  |_| |_   | |   _| |_ / ____ \| |____  | |___| |__| | |\  | |__| _| |_   | |   _| || |__| | |\  |____) |
# |_____|_| \_|_____|  |_|  |_____/_/    \_|______|  \_____\____/|_| \_|_____|_____|  |_|  |_____\____/|_| \_|_____/ 
                                                                                                                    
#------------------------------------------------------------------------------------------------------------------------

#------------------
#ENGINE GEOMETRY
#------------------

#Making an array for the radi (radaii?) across the engine 
RadiusArr = IP.RadiusArr 

#Chamber information 
Diameter_Chamber = RadiusArr[0] * 2
Circum_Chamber = np.pi * Diameter_Chamber
Area_Chamber = np.pi * ( Diameter_Chamber  / 2 )**2

#Throat information 
Radius_Throat = min(RadiusArr)
Diameter_Throat = Radius_Throat*2
Area_Throat = np.pi * Radius_Throat**2

#Exit Information
Radius_Exit = RadiusArr[36]
Diameter_Exit = Radius_Exit*2
Area_Exit = np.pi * Radius_Exit

#General geometry values
#Contraction ratio 
fac_CR = Area_Chamber / Area_Throat

#eps?
eps = Area_Exit / Area_Throat

#the DX that will be used to iterate through chamber
dx = IP.dx

#Mixture Ratio 
MR = IP.MR

#Designed Chamber
Pc = IP.Pchamber

#Flow Rate Stuff
MassFlowRate = IP.MassFlowRate
FuelFlowRate = IP.FuelFlowRate
OxFlowRate = IP.OxFlowRate



#----------------------------
#COOLING CHANNEL GEOMETRY
#----------------------------
#Assuming Uniform Channel Geometry (i.e the channel geometric properties don't change as we move along the chamber)

#Just hard coding some random placement values for right now. Need to figure out how to either study different geometries or something else 
height_c =  0.00762 #0.3 in
base_c = 0.00508 #0.2 in
rib_width_c = 0.00508 /2 #0.1in
tw = 0.00254 #0.1 in
Afin = base_c * height_c

#Thermal Conductivity of Copper
kc = 398 #W/mK

#Thermal Conductivity of Wall
kw = kc

#Area of coolant channel
A_coolant = base_c * height_c

#Coolant Channel Perimeter
Peri_coolant = height_c*2 + base_c*2


#Hydrualic Diameter
#Ac: Area
#Sc: Perimeter of Cooling Channel
dh = TM.dh(A_coolant, Peri_coolant)


#THIS IS NOOOOT NOT NOT GOING TO GIVE A WHOLE NUMBER SO I GOTTA FIGURE THIS SHIT OUT HOW TO GET A WHOLE NUMBER AND STUFF BUT FOR  NOW FORGET IT 
N = Circum_Chamber / ( base_c + rib_width_c)


Channel_Flow_Rate = FuelFlowRate / N



#----------------------------
#CEA DATA
#----------------------------

#Setting up initial CEA call
ispObj = CEA_Obj( oxName = 'N2O', fuelName = 'Isopropanol')


#returns the tuple (mw, gam) for the chamber (lbm/lbmole, N/A)
gammaArr = ispObj.get_Chamber_MolWt_gamma( Pc=Pc, MR=MR)
gamma = gammaArr[1]

#list of heat capacity, viscosity, thermal conductivity and Prandtl number at the exit
visc_gArr = ispObj.get_Exit_Transport(Pc = Pc, MR = MR)

#SPECIFIC HEAT 
#Specific Heat of Gas (Cp)[Need to convert from BTU/LbR to J/kgK]
Cp_g = visc_gArr[0] * 4.18680000000869

#VISCOSITY
#I belive this may be in millipoise, going to convert it to Poise 
visc_g = visc_gArr[1] / 1000

#THERMAL CONDUCTIVITY
#Thermal Conductivity (k), units are mcal/cm-K-s (EW)
Thermal_Conductivity_g = visc_gArr[2] * 4186.8

#PRANDTL NUMBER
Pr_g = visc_gArr[3]


#GETTING DENSITIES
Density_gArr = ispObj.get_Densities(Pc = Pc, MR = MR)
#Need to convert from lb/ft^3 to kg/m^3
Denisty_gExit = Density_gArr[2] * 16.018


#Cstar is in ft/s need to convert to m/s
Cstar = ispObj.get_Cstar( Pc=Pc, MR=MR) / 3.281

#Exit Mach Number
M_e = ispObj.get_MachNumber( Pc = Pc, MR = MR ) 

#Setting up Initial Temperature of gas at Nozzle Exit                                                                                                                    
Taw = IP.TwcArr[36] #K

#Chamber Enthalpy and then convert from BTU/lbm to J/kg
H_chamber = ispObj.get_Chamber_H( Pc = Pc, MR = MR ) * 2326



#--------------------------------------------------------------------
# COOLANT INFORMATION AND COOLANT INFORMATION FROM COOLPROP
#--------------------------------------------------------------------

#Initial coolant temperature as it enters nozzle (room temp)
Tcoolant = 298 #K

#Guess of initial Coolant Pressure. This will obviously change after I do pressure drop calculations but for now forget it
Pcoolant = Pc + (0.5 * Pc) 

#Finding density of Methanal
density_coolant_initial = PropsSI('D', 'T', Tcoolant, 'P', Pcoolant, 'METHANOL' )


Pr_coolant = PropsSI('PRANDTL', 'T', Tcoolant, 'P', Pcoolant, 'METHANOL' )

Visc_coolant = PropsSI('V', 'T', Tcoolant, 'P', Pcoolant, 'METHANOL')

Cp_c = PropsSI('C', 'P', Pcoolant, 'T', Tcoolant, 'METHANOL')

k_c = PropsSI('L', 'T', Pcoolant, 'P', Tcoolant, 'METHANOL')










#------------------------------------------------------------------------------------------------------------------------
#    (                                                                          
#    )\ )           )        (      (              (          )                 
#   (()/(     (  ( /((     ) )\     )\             )\ ) (  ( /((                
#    /(_))(   )\ )\())\ ( /(((_)  (((_)  (   (    (()/( )\ )\())\  (   (    (   
#   (_))  )\ |(_|_))((_))(_))_    )\___  )\  )\ )  ((_)|(_|_))((_) )\  )\ ) )\  
#   |_ _|_(_/((_) |_ (_|(_)_| |  ((/ __|((_)_(_/(  _| | (_) |_ (_)((_)_(_/(((_) 
#    | || ' \)) |  _|| / _` | |   | (__/ _ \ ' \)) _` | | |  _|| / _ \ ' \)|_-< 
#   |___|_||_||_|\__||_\__,_|_|    \___\___/_||_|\__,_| |_|\__||_\___/_||_|/__/
#(Initial Conditions)
# #------------------------------------------------------------------------------------------------------------------------
#                                                                                         

#THIS IS GOING TO BE A GUESS
Tw = 2000 #k            

#-------------------------------------------------------------------------------------------------------------------------------------------
#        ___  __   __    ___ _  _ __     __ ____ __ __ _  ___       _  _  ___ 
#       / __)/ _\ (  )  / __) )( (  )   / _(_  _|  |  ( \/ __)     / )( \/ __)
#      ( (__/    \/ (_/( (__) \/ ( (_/\/    \)(  )(/    ( (_ \     ) __ ( (_ \
#       \___)_/\_/\____/\___)____|____/\_/\_(__)(__)_)__)\___/     \_)(_/\___/
# (Calculating INITIAL Hg)
#-------------------------------------------------------------------------------------------------------------------------------------------
#For the beta calculation going to assume stagnation temperature (To)is the same as the chamber temperature 
beta = TM.beta(Tw, Taw, gamma, M_e)

#Calculating initialy velocity of coolant
V_coolant = FE.FluidVelocity(density_coolant_initial, Channel_Flow_Rate, A_coolant)

#Finding hg initial at Exit 
hg = TM.hg(Diameter_Throat, Denisty_gExit ,Cp_g, Pr_g, Pc ,Cstar, Area_Throat, Area_Exit, beta)



#-------------------------------------------------------------------------------------------------------------------------------------------
#         ___  __   __    ___ _  _ __     __ ____ __ __ _  ___       _  _  ___ 
#        / __)/ _\ (  )  / __) )( (  )   / _(_  _|  |  ( \/ __)     / )( \/ __)
#       ( (__/    \/ (_/( (__) \/ ( (_/\/    \)(  )(/    ( (_ \     ) __ ( (__ 
#        \___)_/\_/\____/\___)____|____/\_/\_(__)(__)_)__)\___/     \_)(_/\___)
#-------------------------------------------------------------------------------------------------------------------------------------------

Pr_c = FE.Pr(Cp_c, Visc_coolant, k_c)
Red_c = FE.Red(density_coolant_initial, V_coolant, height_c, Visc_coolant)

hc = TM.hc(Red_c, Pr_c, k_c, dh)



#-------------------------------------------------------------------------------------------------------------------------------------------
#        _  _ ____  __ ____    ____ ____  __   __ _  ____ ____ ____ ____     __   ___ ____  __  ____ ____    _  _  __   __    __   
#       / )( (  __)/ _(_  _)  (_  _|  _ \/ _\ (  ( \/ ___|  __|  __|  _ \   / _\ / __|  _ \/  \/ ___) ___)  / )( \/ _\ (  )  (  )  
#       ) __ () _)/    \)(      )(  )   /    \/    /\___ \) _) ) _) )   /  /    ( (__ )   (  O )___ \___ \  \ /\ /    \/ (_/\/ (_/\
#       \_)(_(____)_/\_(__)    (__)(__\_)_/\_/\_)__)(____(__) (____|__\_)  \_/\_/\___|__\_)\__/(____(____/  (_/\_)_/\_/\____/\____/
#-------------------------------------------------------------------------------------------------------------------------------------------
#Here I am unsure if I should use the disseration approach or the approach done in Kose and Celik


#Kose and Celik:

#Fin Efficiency
fineff2 = TM.fin_eff(hc, rib_width_c, kc, height_c)

hc_f = TM.hc_f(hc, base_c, fineff2, height_c, rib_width_c)

#Heat Flux
q = TM.q(Taw, Tcoolant, hg, tw, kw, hc_f)

#Heat Flux from Outer Wall of Engine to Coolant
q_ow_c = TM.q_ow_c(hc_f, Tw, Tcoolant)

Two = TM.Two(hc, Taw, Tcoolant, hg, tw, kw, hc_f)

TempHotWall = TM.TempHotWall(tw, kw, Taw, Tcoolant, hg, hc_f, Two)

#Heat Flux from hot combustion gases to inner wall of thrust chamber
#q_hw_iw = TM.q_hg_iw(hg, Taw, Tw)

#Heat Flux from inner combustion chamber wall to base of cooling channel, Two
#q_iw_wo = TM.q_iw_wo(kw, tw, Tw, Twc)





#DISSERTATUION APPROACH :
m = TM.m(hc, rib_width_c, 0.1, kc)
fineff = TM.fineff(m, height_c)

qfin = TM.qfin(hc, Tcoolant, Tcoolant, fineff)


#h_hw: static enthalpy of hot wall
#h_aw: 
Tcw = TM.Tcw(hg, (dx * base_c), H_chamber, 0, hc, (dx * base_c), qfin, Afin, Tcoolant)

Thw = TM.Thw(hg, (dx * base_c), kc, (dx * base_c), tw, tw, 0, H_chamber, Tcw)








#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#  _______        _              _                  _ _   _               
# |__   __|      | |       /\   | |                (_) | | |              
#    | | ___  ___| |_     /  \  | | __ _  ___  _ __ _| |_| |__  _ __ ___  
#    | |/ _ \/ __| __|   / /\ \ | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \ 
#    | |  __/\__ \ |_   / ____ \| | (_| | (_) | |  | | |_| | | | | | | | |
#    |_|\___||___/\__| /_/    \_\_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_|
#                                   __/ |                                 
#                                  |___/                                  
#-------------------------------------------------------------------------------------------------------------------------------------------


#Initial Wall Temp Guess
Tw_new = Taw - 10  #

#Setting initial Coolant Temperature and Pressure, Density and Viscosity for the loop
#They are set as the looping variables cause they will change as we loop 
Tc_j1 = Tcoolant
P_coolantj1 = Pcoolant
density_new = density_coolant_initial
Visc_new = Visc_coolant

Tw_tolerance = 10#0.05 * Tw_new
Pc_tolerance = 10#0.05 * P_coolantj1
visc_tolerance = 100
density_tolerance = 100

tolerance = 0.0005

NumberOfDataPoints = 9
LocationArr = IP.LocationArr

DataArr=  np.array([[0] * NumberOfDataPoints] * len(LocationArr))

i = 0
n = 36





for j in np.flip(IP.LocationArr) : #np.flip(IP.LocationArr):
    #Finding Temperature in the part of the engine we are at. Should be updated later on with hand calcs but for now using RPA
    Taw = IP.TwcArr[n]

    #Finding Chamber Gas Properties at different points in the engine. Should be updated at a later date with hand calcs
    if IP.LocationArr[n] > IP.Lthroat:
        #In Nozzle So...
        MachNumber = ispObj.get_MachNumber( Pc = Pc, MR = MR, eps = eps )
        gammaArr = ispObj.get_exit_MolWt_gamma(Pc = Pc, MR = MR)
        gamma = gammaArr[1]

        visc_gArr = ispObj.get_Exit_Transport(Pc = Pc, MR = MR)
        #Specific Heat of Gas (Cp)[Need to convert from BTU/LbR to J/kgK]
        Cp_g = visc_gArr[0] * 4.18680000000869
        #I belive this may be in millipoise, going to convert it to Poise 
        visc_g = visc_gArr[1] / 1000
        #Thermal Conductivity (k), units are mcal/cm-K-s (EW)
        Thermal_Conductivity_g = visc_gArr[2] * 4186.8
        #PRANDTL NUMBER
        Pr_g = visc_gArr[3]

        
    if IP.LocationArr[n] == IP.Lthroat:
        #In Throat
        MachNumber = 1
        gammaArr = ispObj.get_Throat_Molwt_gamma(Pc = Pc, MR = MR)
        gamma = gammaArr[1]

        visc_gArr = ispObj.get_Throat_Transport(Pc = Pc, MR = MR)
        #Specific Heat of Gas (Cp)[Need to convert from BTU/LbR to J/kgK]
        Cp_g = visc_gArr[0] * 4.18680000000869
        #I belive this may be in millipoise, going to convert it to Poise 
        visc_g = visc_gArr[1] / 1000
        #Thermal Conductivity (k), units are mcal/cm-K-s (EW)
        Thermal_Conductivity_g = visc_gArr[2] * 4186.8
        #PRANDTL NUMBER
        Pr_g = visc_gArr[3]       

    else:
        #We're in the Chamber
        MachNumber = ispObj.get_Chamber_MachNumber(Pc = Pc, MR = MR, fac_CR = fac_CR )
        gammaArr = ispObj.get_Chamber_MolWt_gamma(Pc = Pc, MR = MR)
        gamma = gammaArr[1]

        visc_gArr = ispObj.get_Chamber_Transport( Pc = Pc, MR = MR)
        #Specific Heat of Gas (Cp)[Need to convert from BTU/LbR to J/kgK]
        Cp_g = visc_gArr[0] * 4.18680000000869
        #I belive this may be in millipoise, going to convert it to Poise 
        visc_g = visc_gArr[1] / 1000
        #Thermal Conductivity (k), units are mcal/cm-K-s (EW)
        Thermal_Conductivity_g = visc_gArr[2] * 4186.8
        #PRANDTL NUMBER
        Pr_g = visc_gArr[3]              



    #Passing in above information as the first input for the loop in this section of channel cooling: 
    Tw_new = IP.TwcArr[n] #K
    Radius = IP.RadiusArr[n] #m
    Area = np.pi * Radius ** 2 #m^2

    Tw_tolerance = 0.05 * Tw_new
    tolerance = 0.00005
    T_coolant_i = Tc_j1
    P_Coolant = P_coolantj1



    while Tw_tolerance > tolerance:
        #print('')
        #print('')
        print('Has not converged yet')

        #Set initial parameters to ones that were calculated before
        Tw = Tw_new 
        T_coolant = Tc_j1
        print('Old Tcoolant that was calculated last step and used now is: %g'%T_coolant)
        Pcoolant = P_coolantj1


        beta = TM.beta(Tw, Taw, gamma, M_e)

        hg = TM.hg(Diameter_Throat, Denisty_gExit ,Cp_g, Pr_g, Pc ,Cstar, Area_Throat, Area, beta)

        density_coolant = density_new #PropsSI('D', 'T', Tcoolant, 'P', Pcoolant, 'METHANOL' )
        Visc_coolant = Visc_new #PropsSI('V', 'T', Tcoolant, 'P', Pcoolant, 'METHANOL')
        Cp_c = PropsSI('C', 'T', T_coolant, 'P', Pcoolant, 'METHANOL')
        print('Cp_c: %g'%Cp_c)
        k_c = PropsSI('L', 'T', T_coolant, 'P', Pcoolant, 'METHANOL')

        V_coolant = FE.FluidVelocity(density_coolant_initial, Channel_Flow_Rate, A_coolant)
        Pr_c = FE.Pr(Cp_c, Visc_coolant, k_c)
        Red_c = FE.Red(density_coolant_initial, V_coolant, height_c, Visc_coolant)

        hc = TM.hc(Red_c, Pr_c, k_c, dh)
        
        #print('Calculated hc: %g'%hc)

        #Fin Efficiency
        fineff2 = TM.fin_eff(hc, rib_width_c, kc, height_c)

        hc_f = TM.hc_f(hc, base_c, fineff2, height_c, rib_width_c)

        #Heat Flux
        #Coolant here needs to be updated too 
        q = TM.q(Taw, T_coolant, hg, tw, kw, hc_f)
        print('q: %g'%q)

        #Calculate new Twall 
        Tw_new = Taw - ( q / hg ) #Tw-0.01 

        Tw_tolerance = ( abs( Tw - Tw_new ) ) / ( Tw )

        while density_tolerance and visc_tolerance > tolerance:

            #Not sure if height_c or diameter of chamber at that point
            Tc_j1 = TM.Tc_j1(T_coolant, Radius*2, dx, q, N, FuelFlowRate, Cp_c)
            print('Tc_j1: %g'%Tc_j1)

            e = 0.045
            f = TM.f_j(Red_c, e, height_c)

            Kl_j = 0.08
            P_coolantj1 = TM.Pc_j1(Pcoolant, density_coolant, V_coolant, Kl_j, f, dx, height_c)

            Pc_tolerance = ( abs( Pcoolant - P_coolantj1 ) ) / ( Pcoolant )

            #print('Pressure coolant: %g'%P_coolantj1)
            #Method below coming from paper 
            #Calculate fluid properties with Enthalpy and Pressure


            #Not sure if I need to use the calculated H+1 from the calculation above, or the H from cool prop
            Hcoolant = PropsSI('H','P',P_coolantj1,'T', Tc_j1,'METHANOL')
            density_new = PropsSI('D', 'H', Hcoolant, 'P', P_coolantj1, 'METHANOL' )
            Visc_new = PropsSI('V', 'H', Hcoolant, 'P', P_coolantj1, 'METHANOL')
            #print('New Density: %g'%density_new)
            #print('New Viscosity: %g'%Visc_new)


        i = i + 1

        H_Out = Hcoolant
        Density_Out = density_new
        Visc_Out = Visc_new
        P_Coolant_Out = P_coolantj1
        Temp_Out = Tc_j1
        Temp_Wall = Tw_new
        q = q
        print(i)
        #print('Taw: %g'%Taw)



    DataArr[n] = [H_Out, Density_Out, Visc_Out, P_Coolant_Out, Temp_Out, Temp_Wall, q, Tc_j1, Cp_c]
    n = n - 1




print('hc out:                   %g'%hg)
print('Denisty Out:              %g'%Density_Out)
print('Viscosity Out:            %g'%Visc_new)
print('Pressure of Coolant out:  %g'%P_Coolant_Out)
print('Coolant Inlet Temp:       %g'%Tcoolant)
print('Temp of Coolant Out:      %g'%Temp_Out)
print('Temp Wall:                %g'%Tw_new)
print('Number of Iterations:     %g'%i)

print('[H_Out, Density_Out, Visc_Out, P_Coolant_Out, Temp_Out, Temp_Wall]')
print(DataArr)


"""
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
fig.suptitle('Cooling Channel Study Results')

ax1.plot(IP.LocationArr, DataArr[:, 5], label='Wall Temperatures')
ax1.set_xlabel('Chamber Location (m)')
ax1.set_ylabel('Wall Temperature')
ax1.grid(color='r', linestyle='-', linewidth=0.1)

ax2.plot(IP.LocationArr, DataArr[:, 6], label='q')
ax2.set_xlabel('Chamber Location')
ax2.set_ylabel('Heat Flux')
ax2.grid(color='g', linestyle='-', linewidth=0.1)

ax3.plot(IP.LocationArr, IP.TwcArr, label = 'Chamber Temperatures')
ax3.set_xlabel('Chamber Location (m)')
ax3.set_ylabel('Chamber Temperature (K)')
ax3.grid(color = 'b', linestyle= '-', linewidth = 0.1)

ax4.plot(IP.LocationArr, DataArr[:, 7], label = 'Coolant Temperature')
ax4.set_xlabel('Chamber Location (m)')
ax4.set_ylabel('Coolant Temperature')
ax4.grid(color = 'g', linestyle = '-', linewidth = 0.1)


plt.show()


"""

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.suptitle('What Variable go into Channel Cooling and What the Fuck is going on with them')

ax1.plot(IP.LocationArr, DataArr[:, 6], label='q')
ax1.set_xlabel('Chamber Location (m)')
ax1.set_ylabel('q')
ax1.grid(color='r', linestyle='-', linewidth=0.1)

ax2.plot(IP.LocationArr, DataArr[:, 8], label='Cp_c')
ax2.set_xlabel('Chamber Location (m)')
ax2.set_ylabel('Cp_c')
ax2.grid(color='r', linestyle='-', linewidth=0.1)

ax3.plot(IP.LocationArr, DataArr[:, 7], label='Coolant Temperatures')
ax3.set_xlabel('Chamber Location (m)')
ax3.set_ylabel('Coolant Temperature')
ax3.grid(color='r', linestyle='-', linewidth=0.1)

plt.show()