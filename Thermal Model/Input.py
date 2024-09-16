import pandas as pd
import numpy as np 





#--------------------------------------------------------
#ENGINE GEOMETRY AND INFORMATION FROM SIZING PROGRAM
#--------------------------------------------------------

MR = 6
Pc  = 300 #PSI
Pchamber = Pc * 6895 #Pa

MassFlowRate = 1.08 #kg/s 
FuelFlowRate = MassFlowRate / ( 1 + MR )
OxFlowRate = FuelFlowRate * MR

ChamberDiameter = 0.08326 #m
ChamberCirc = np.pi * ChamberDiameter #m


Lcyl = 0.15159 #m
Lc = 0.2112 #m 
Le = 64.47 #m

Lthroat = Lc - Lcyl #m




#Getting the information from the excel file that has the RPA thermal data on it
RPAThermalArr = pd.read_excel("ThermalAnalysis.xlsx")


#Making Raidus and Length Arrays for Later 
RadiusArr = np.array(RPAThermalArr["Radius (mm)"]) / 1000
LocationArr = np.array(RPAThermalArr["Location (mm)"]) / 1000



dx = LocationArr[36] / len(LocationArr)

TwcArr= np.array(RPAThermalArr["Twc (K)"])



NumberOfDataPoints = 5
DataArr=  np.array([[0] * NumberOfDataPoints] * len(LocationArr))

