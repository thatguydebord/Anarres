import numpy as np 
import CoolProp as CP
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj
import colebrook 



#               ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓███████▓▒░ 
#               ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
#               ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
#               ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
#               ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
#               ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
#               ░▒▓█▓▒░       ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░  
                                                                                                               
                                                                                                               





#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#    ___ ___                __    ___________                             _____              ___________                   __  .__                      
#   /   |   \  ____ _____ _/  |_  \__    ___/___________    ____   ______/ ____\___________  \_   _____/_ __  ____   _____/  |_|__| ____   ____   ______
#  /    ~    \/ __ \\__  \\   __\   |    |  \_  __ \__  \  /    \ /  ___|   __\/ __ \_  __ \  |    __)|  |  \/    \_/ ___\   __\  |/  _ \ /    \ /  ___/
#  \    Y    |  ___/ / __ \|  |     |    |   |  | \// __ \|   |  \\___ \ |  | \  ___/|  | \/  |     \ |  |  /   |  \  \___|  | |  (  <_> )   |  \\___ \ 
#   \___|_  / \___  >____  /__|     |____|   |__|  (____  /___|  /____  >|__|  \___  >__|     \___  / |____/|___|  /\___  >__| |__|\____/|___|  /____  >
#         \/      \/     \/                             \/     \/     \/           \/             \/             \/     \/                    \/     \/ 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#ALL EQUATIONS BELOW COME FROM: 
    #Regenerative Cooling Comparison of LOX/LCH4 and LOX/LC3H8 Rocket Engines Using the One-Dimensional Regenerative Cooling Model Tool ODREC


#Taw: Adiabatic Wall Temperature 
#To: Stagnation Temperature
def Taw(To, Pr, gamma, M):
    return To * ( ( 1 + Pr**(1/3) * ( ( gamma - 1 ) / 2 ) * M**2 ) / ( 1 + ( ( gamma - 1 ) / 2 ) * M**2 ) )


#Convection Coefficient for Hot Gas
def hg(dt, ug, Cp_g, Pr_g, P_cc, Cstar, At, A, beta):
    return ( 0.026 / (dt**0.2) ) * ( ( ug**0.2 * Cp_g ) / ( Pr_g**0.6 ) ) * ( ( P_cc / Cstar )**0.8 ) * ( ( At / A )**0.9 ) * beta


#Correction Factor
#Tw: Wall Temperatures on hot gas side 
def beta(Tw, To, gamma, M):
    return ( ( 0.5 * ( Tw/ To ) * ( 1 + ( ( gamma - 1 ) / 2 ) * M**2 ) + 0.5 )**(-0.68) ) * ( ( 1 + ( ( gamma - 1 ) / 2 ) * M**2 )**(-0.12) )


#Heat Flux from hot combustion gases to inner wall of thrust chamber
def q_hg_iw(hg, Taw, Tw):
    return hg * ( Taw - Tw )


#Heat Flux from inner combustion chamber wall to base of cooling channel, Two
def q_iw_wo(kw, tw, Tw, Two):
    return ( kw / tw ) * ( Tw - Two ) 


#Hydrualic Diameter
#Ac: Area
#Sc: Perimeter of Cooling Channel
def dh(Ac, Sc):
    return ( 4 * Ac ) / Sc


#Convection Coefficient for Coolant Flow 
#Dittus-Boetler Correlation
#kc: Thermal Conductivity of Coolant
#dh: Hydrualic Diameter
def hc(Re, Pr, kc, dh):
    return ( 0.023 * Re**0.8 * Pr**0.4 * kc ) / dh


#Fin Efficiency 
#wb: Fin Width 
def fin_eff(hc, wb, kw, height):
    return np.tanh( np.sqrt( ( 2 * hc * wb ) / ( kw ) ) * ( height / wb ) ) / ( np.sqrt( ( 2 * hc * wb ) / ( kw ) ) * ( height / wb ) )


#Convection Coefficient for Fin 
#w: Channel Width
#wb: Fin Width 
def hc_f(hc, w, fin_eff, height, wb):
    return hc * ( ( w + 2 * fin_eff * height ) / ( w + wb ) )


#Heat Flux from Outer Wall of Engine to Coolant
def q_ow_c(hc_f, Tw_o, Tc):
    return hc_f * ( Tw_o - Tc ) 



def q(Taw, Tc, hg, tw, kw, hc_f):
    return ( Taw - Tc ) / ( (1/hg) + (tw/kw) + (1/hc_f) )


def Two(hc, Taw, Tc, hg, tw, kw, hc_f):
    return ( ( 1 / hc ) * ( ( Taw - Tc ) / ( (1/hg) + (tw/kw) + (1/hc_f) ) ) ) + Tc


def TempHotWall(tw, kw, Taw, Tc, hg, hc_f, Two):
    return ( tw / kw ) * ( ( Taw - Tc ) / ( ( 1 / hg ) + ( tw / kw ) + ( 1 / hc_f ) )) + Two


#----------------------------------------------------------------------------------------------------
#TAKEN FROM THAT DISSERTATION
#Asm_hw: Surface Area of the Hot Wall Section you are calculating 
#Asm_cw: Surface Area of Cold Wall Section you are calculating
#qdashdashfin: q" heat flux of fin 
#Haw: Static Enthalpy of 
#Hhw: Static Enthalpy of Hot Wall 
#m: Intrmdiat calc pramtr
#hgt: Hight

def m(hc, wfin, dx, km):
    return ( 2 * hc * ( wfin + dx ) ) / ( km * wfin * dx) 


def fineff(m, hgt):
    return ( np.tanh(np.deg2rad( m * hgt )) ) / ( m * hgt )
    r

def qfin(hc, Tcw, Tc, fineff):
    return hc * ( Tcw - Tc ) * fineff 


def Tcw(hg, Asm_hw, Haw, Hhw, hc, Asm_cw, qdashdashfin, Afin, Tcool):
        return ( (hg * Asm_hw * ( Haw - Hhw ) - qdashdashfin * Afin) / ( hc * Asm_cw ) ) + Tcool

def Thw(hg, Asm_hw, km_i, Asm, t, tlayer, Haw, Hhw, Tcw):
    return ( ( ( hg * Asm_hw * km_i * Asm ) / ( t * tlayer ) ) ) * ( Haw - Hhw )  + Tcw


def Hj1(massflowrate, q, Hi, density, Area):
    return ( 1 / massflowrate ) * ( q + massflowrate * ( ( massflowrate ** 2 ) / ( 2 * density**2 * Area**2 ) ) )

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#       _____  .__                      .__  __  .__             
#     /  _  \ |  |   ____   ___________|__|/  |_|  |__   _____  
#    /  /_\  \|  |  / ___\ /  _ \_  __ \  \   __\  |  \ /     \ 
#  /    |    \  |_/ /_/  >  <_> )  | \/  ||  | |   Y  \  Y Y  \
#  \____|__  /____|___  / \____/|__|  |__||__| |___|  /__|_|  /
#        \/    /_____/                            \/      \/ 
#                                                            
#   Energy flows into each control volume by heat transfer from hot gas to walls and through the enthalpy of entering coolant fluid
#   Energy flows out of each control volume through enthalpy of exititing coolant fluid 

#   At each section, heat flux is calculated by function 'q' above USING the temperature of the coolant calculated at the previous section by Tc_j1

#   At each section, wall temperature is iterated until a convergence criterion is satisfied
#-----------------------------------------------------------------------------------------------------------------------------------------------------------


#Coolant Temperature for Control Volume j 
#L: Length of particular Section
#j denotes section index
#Total coolant mass flow rate
def Tc_j1(Tc_j, di_j, L_j, q_j, N, massflowrate, Cp_c):
    return Tc_j + ( ( np.pi * di_j * L_j * q_j ) / ( N * massflowrate * Cp_c ) )

#Inner Wall 
def Tw_j(Taw_j, q_j, hg_j):
    return Taw_j - ( q_j / hg_j )

#NEED TO INPUT TWC AND THW FROM OTHER PAPER HERE 


#Finding the Friction Factor 
def f_j(Re, e, D):
    return colebrook.sjFriction(Re, (e/D))

#Coolant Pressure in each Section
#v is velocity 
#Kl is minor loss coefficient 
def Pc_j1(Pc_j, density_c_j, v_j, Kl_j, f_j, L_j, dh_j):
    return Pc_j - ( ( density_c_j * v_j**2 ) / 2 ) * ( Kl_j + f_j * ( L_j / dh_j ) )


##-----------------------------------------------------------------------------------------------------------------------------------------------------------
#
# 
#   ___  _                  _ _   _                           __      _   _           
# / _ \| |                (_| | | |                         / _|    | | | |          
#/ /_\ | | __ _  ___  _ __ _| |_| |__  _ __ ___        ___ | |_     | |_| |__   ___  
#|  _  | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \      / _ \|  _|    | __| '_ \ / _ \ 
#| | | | | (_| | (_) | |  | | |_| | | | | | | | |    | (_) | |      | |_| | | |  __/ 
#\_| |_|_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_|     \___/|_|       \__|_| |_|\___| 
#          __/ |                                                                     
#         |___/                                                                      
#
#
# _____             _             _       ______ _                                         _ 
#/  __ \           | |           | |      | ___ | |                                       | |
#| /  \/ ___   ___ | | __ _ _ __ | |_     | |_/ | |__   __ _ ___  ___       __ _ _ __   __| |
#| |    / _ \ / _ \| |/ _` | '_ \| __|    |  __/| '_ \ / _` / __|/ _ \     / _` | '_ \ / _` |
#| \__/| (_) | (_) | | (_| | | | | |_     | |   | | | | (_| \__ |  __/    | (_| | | | | (_| |
# \____/\___/ \___/|_|\__,_|_| |_|\__|    \_|   |_| |_|\__,_|___/\___|     \__,_|_| |_|\__,_|
#                                                                                            
#                                                                                            
#
# _____ _                              ______ _               _           _     ______                          _   _          
#|_   _| |                             | ___ | |             (_)         | |    | ___ \                        | | (_)         
#  | | | |__   ___ _ __ _ __ ___   ___ | |_/ | |__  _   _ ___ _  ___ __ _| |    | |_/ _ __ ___  _ __   ___ _ __| |_ _  ___ ___ 
#  | | | '_ \ / _ | '__| '_ ` _ \ / _ \|  __/| '_ \| | | / __| |/ __/ _` | |    |  __| '__/ _ \| '_ \ / _ | '__| __| |/ _ / __|
#  | | | | | |  __| |  | | | | | | (_) | |   | | | | |_| \__ | | (_| (_| | |    | |  | | | (_) | |_) |  __| |  | |_| |  __\__ \
#  \_/ |_| |_|\___|_|  |_| |_| |_|\___/\_|   |_| |_|\__, |___|_|\___\__,_|_|    \_|  |_|  \___/| .__/ \___|_|   \__|_|\___|___/
#                                                    __/ |                                     | |                             
#                                                   |___/                                      |_|                                                                                                                     
#
# 
#   Thermophysical properties of the coolant along the channels are determined by interpolations,
#       with the values in the tables provided by the user, using the pressure and temperature values of the coolant.
#       
#
#   In the liquid phase, the thermophysical property table is provided for a range of temperatures without using any pressure value 
#       since the thermophysical properties of liquids are almost invariant of pressure due to the negligible compressibility.
# 
#
#   At the moment that the phase change begins, the specific enthalpy of the coolant is equal to the saturated liquid’s specific enthalpy. 
#       As the phase changing coolant flows through the channels, its pressure drops at the same time its specific enthalpy rises due to the absorbed heat. 
#       The coolant state will stay on the saturation line until the phase change is completed; 
#           therefore, the coolant temperature will drop in the course of the phase change even if the enthalpy rises.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

#The phase of a coolant at a section is determined using the interpolation temperature that is shown below
#Tsat and Psat are the temps and pressure values on the saturation line, which are read from the saturation thermophysical property table
#subscripts 'u' and 'l' denote 'upper' and 'lower' indicating that the coolant pressure that that particular section is between these values

#If the coolant pressure at the particular section is greater than the highest pressure value on the saturation line,
# then the interpolated temperature becomes the highest temperature value on the saturation line.
def interp_temp(Tsat_l, Tsat_u, Psat_u, Pc, Psat_l):
    return Tsat_l + ( Tsat_u - Tsat_l ) * ( ( Psat_u - Pc ) / ( Psat_u - Psat_l ) )


def hc_j1(hc_j, di_j, L_j, q_j, N, massflowrate):
    return hc_j + ( ( np.pi * di_j * L_j * q_j ) / ( N * massflowrate ) ) 


#hf_int and hg_int are saturated liquid and saturated vapor enthalpies 
def vapor_fraction(hg_int, hc, hf_int):
    return ( hg_int - hc ) / ( hg_int - hf_int )