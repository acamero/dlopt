# https://github.com/RebeccaSalles/TSPred/wiki/EUNITE

load(file="EUNITE.Loads.rda")
load(file="EUNITE.Loads.cont.rda")
load(file="EUNITE.Reg.rda")
load(file="EUNITE.Reg.cont.rda")
load(file="EUNITE.Temp.rda")
load(file="EUNITE.Temp.cont.rda")


#Prepares the input data
#Selects the temperature input data relative to EUNITE.Loads and binds it to the other input variables in EUNITE.Reg
reg <- cbind(tail(EUNITE.Temp, nrow(EUNITE.Loads)),EUNITE.Reg)
regCont <- cbind(EUNITE.Temp.cont, EUNITE.Reg.cont)

#Generates a single time series of the maximum daily electrical loads
MaxLoads <- apply(EUNITE.Loads, 1, max)
#Generates a single time series of the continuation values of MaxLoads which were to be predicted
MaxLoadsCont <- apply(EUNITE.Loads.cont, 1, max)

training_data <- cbind(EUNITE.Loads, MaxLoads, reg)
testing_data <- cbind(EUNITE.Loads.cont, MaxLoadsCont, regCont)

write.csv(training_data, file="eunite.training.csv", sep=",")
write.csv(testing_data, file="eunite.testing.csv", sep="," )
