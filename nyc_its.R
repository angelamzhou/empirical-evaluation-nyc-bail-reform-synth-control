install.packages("tscount")
install.packages("readxl")
library("tscount")
library("readxl")
require("lubridate")
library(data.table)
library(dplyr)
require("ggplot2")
library("forecast")
library("texreg")

df <- read.csv('incident_series_newyork_update.csv')
df[is.na(df)] <- 0 

# If using weekly aggregation 
# df <- read.zoo(df)
# rollapply(df, 7, sum, by = 7)
# df <- rollapply(df, 7, sum, by = 7)
# df <- fortify.zoo(df, name = "Date")
df$date <- df$Date


# If day of week fixed effects 
df <- df %>%
  mutate(date = as.Date(date)) %>%
  mutate(weekday = weekdays(date)) %>%
  dcast( date + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + t   ~ weekday  , fun.aggregate = length)

df <- df %>% 
  mutate(date = as.Date(date)) %>% 
  mutate(month = months(date)) %>% 
  dcast(date + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + Monday+Tuesday+Wednesday+Thursday+Friday+Saturday+ t  ~  month , fun.aggregate = length)

df <- df %>% 
  mutate(date = as.Date(date)) %>% 
  mutate(year = year(date)) %>% 
  dcast(date + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + Monday+Tuesday+Wednesday+Thursday+Friday+Saturday+t + January+February+March+April+May+June+July+August+September+October+November ~ year , fun.aggregate = length)

holidays <- c(as.Date("2017-01-01"), as.Date("2017-05-29"), as.Date("2017-07-04"), as.Date("2017-10-31"), as.Date("2017-12-25"),as.Date("2017-12-31"),
              as.Date("2018-01-01"), as.Date("2018-05-28"), as.Date("2018-07-04"), as.Date("2018-10-31"), as.Date("2018-12-25"),as.Date("2018-12-31"),
              as.Date("2019-01-01"), as.Date("2019-05-27"), as.Date("2019-07-04"), as.Date("2019-10-31"), as.Date("2019-12-25"),as.Date("2019-12-31"),as.Date("2020-01-01")
)

df$holiday <- ifelse(df$date %in% holidays, 1, 0)


categories = c("Violent_Assault","Property_Theft","Property_Burglary","Drug_Drug","Violent_Robbery", "Violent_Homicide")
df_cats <- df[, c(categories, "date")]
df_cats$date <- as.Date(df_cats$date)
pre <- df_cats[df_cats$date < "2020-01-01",categories]
post <- df_cats[df_cats$date > "2020-01-01",categories]

xnames = c(
"t",
"Monday",
"Tuesday",
"Wednesday",
"Thursday",
"Friday",
"Saturday",
"January",
"February",
"March",
"April",
"May",
"June",
"July",
"August",
"September",
"October",
"November",
"2017",
"2018",
"2019",
"holiday"
)

df[["Violent_Homicide"]]

xreg = df[, xnames]


res_fittedvalues <- data.frame(matrix(ncol=length(categories),nrow=dim(df)[1]))
res_coefs <- list()
res_arimas <- list() 
p_vals_ <- list()


i<-0
for (category in categories){
  i<-i+1
  y <- df[[category]]
  aa <- auto.arima(df[[category]],xreg=as.matrix(xreg))
  p_vals <- (1-pnorm(abs(aa$coef)/sqrt(diag(aa$var.coef))))*2
  p_vals["t"]
  p_vals_[[category]] <- p_vals 
  res_fittedvalues[[category]] <- aa$fitted
  res_arimas[[category]] <- aa
  plot(y, cex = 0.3)
  lines(res_fittedvalues[[category]],col="red")
}

for (category in categories){
  p_df <- data.frame(p_vals_[[category]])
  colnames(p_df) <- c("p")
  fn <- paste(category,"-pval.csv",sep="")
  write.csv(p_df, fn)
  coef_ <- data.frame(res_arimas[[category]]$coef)
  fn <- paste(category,"-coef.csv",sep="")
  write.csv(coef_, fn)
}

res_coefs = list() 
for (category in categories){
  res_coefs[[category]] <- res_arimas[[category]]$coef
}

extract.autoarima <- function(model) {
  s <- summary(model)
  names <- names(model$coef)
  co <- as.numeric(model$coef)
  se <- diag(model$var.coef)
  pval <- (1-pnorm(abs(co)/sqrt( se )))*2
  y <- model$residuals + model$fitted
  mu_y <- mean(y)
  
  rs <- 1 - (sum((model$residuals)^2) / sum((y-mu_y)^2))
  
  p <- length(model$coef)
  n <- nobs(model)
  adj <- 1- (1-rs) * (n-1)/(n-p-1)
  
  gof <- c(rs, adj, n)
  gof.names <- c("R$^2$", "Adj.\\ R$^2$", "Num.\\ obs.")
  
  tr <- createTexreg(
    coef.names = names,
    coef = co,
    se = se,
    pvalues = pval,
    gof.names = gof.names,
    gof = gof
  )
  return(tr)
}

print_cats = c("Violent Assault","Property Theft","Drug","Property Burglary","Robbery")

# synth control category extraction 
tr_Drug <- extract.autoarima(res_arimas$Drug_Drug)
tr_Burglary <- extract.autoarima(res_arimas$Property_Burglary)
tr_Theft <- extract.autoarima(res_arimas$Property_Theft)
tr_Violent_Assault <- extract.autoarima(res_arimas$Violent_Assault)
tr_Violent_Robbery <- extract.autoarima(res_arimas$Violent_Robbery)
texreg(c(tr_Violent_Assault, tr_Theft, tr_Burglary, tr_Drug,tr_Violent_Robbery), custom.model.names = print_cats)


setMethod("extract", definition = extract.autoarima)

write.table(res_fittedvalues, paste("fitted_values.csv"))

autoar <- auto.arima(df[[category]],xreg=as.matrix(xreg))

#################
# Homicide ITS 

df <- read.csv('incident_series_newyork_update.csv')
df[is.na(df)] <- 0 

# If using weekly aggregation 
df <- read.zoo(df)
rollapply(df, 7, sum, by = 7)
df <- rollapply(df, 7, sum, by = 7)
df <- fortify.zoo(df, name = "Date")
df$t <- as.integer(df$t > 0)

# If day of week fixed effects 
df$date <- df$Date

df <- df %>% 
  mutate(date = as.Date(date)) %>% 
  mutate(month = months(date)) %>% 
  dcast(date + Violent_Homicide + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + t  ~  month , fun.aggregate = length)

df <- df %>% 
  mutate(date = as.Date(date)) %>% 
  mutate(year = year(date)) %>% 
  dcast(date + Violent_Homicide + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + t + January+February+March+April+May+June+July+August+September+October+November ~ year , fun.aggregate = length)

# treatment X date interaction 
acf(df[[category]],lag.max = 20)
pacf(df[[category]],lag.max = 20)
# MA_lags = c(2, 2, 3, 2, 1)
# AR_lags = c(3, 1, 2, 1, 1)

df$days <- as.integer(difftime(df$date,as.Date("2017-01-05"),unit="days")+1)


xnames = c(
  "t",
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "2017",
  "2018",
  "2019"
)


category = "Violent_Homicide" 
y <- df$Violent_Homicide
xreg = df[, xnames]
ts_arima <- tsglm(y, model=list(past_obs=1, past_mean=2),xreg=xreg, distr="poisson")

res_coefs[[category]] <- ts_arima$coefficients
res_fittedvalues[[category]] <- ts_arima$fitted.values
res_arimas[[category]] <- ts_arima
pdf("homicide-its.pdf") 
plot(df$date,y, cex = 0.3, xlab='date',ylab='# homicides/week')
lines(df$date,ts_arima$fitted.values,col="black", lwd=1.5)
abline(v = as.Date("2020-01-01"), col="black", lty="dotted")
dev.off() 

extract.tsglmarima <- function(model) {
  s <- summary(model)
  names <- names(model$coefficients)
  co <- as.numeric(model$coefficients)
  se <- se(ts_arima)$se
  pval <- (1-pnorm(abs(co)/sqrt( se )))*2
  y <- model$residuals + model$fitted.values
  mu_y <- mean(y)
  
  rs <- 1 - (sum((model$residuals)^2) / sum((y-mu_y)^2))
  
  p <- length(model$coef)
  n <- model$n_obs
  adj <- 1- (1-rs) * (n-1)/(n-p-1)
  
  gof <- c(rs, adj, n)
  gof.names <- c("R$^2$", "Adj.\\ R$^2$", "Num.\\ obs.")
  
  tr <- createTexreg(
    coef.names = names,
    coef = co,
    se = se,
    pvalues = pval,
    gof.names = gof.names,
    gof = gof
  )
  return(tr)
}


setMethod("extract", definition = extract.tsglmarima)
texreg(extract.tsglmarima(ts_arima))

########
# Homicide ITS: dynamic its specification 


df <- read.csv('incident_series_newyork_update.csv')
df[is.na(df)] <- 0 

# If using weekly aggregation 
df <- read.zoo(df)
rollapply(df, 7, sum, by = 7)
df <- rollapply(df, 7, sum, by = 7)
df <- fortify.zoo(df, name = "Date")
df$t <- as.integer(df$t > 0)

# If day of week fixed effects 
df$date <- df$Date

df <- df %>% 
  mutate(date = as.Date(date)) %>% 
  mutate(month = months(date)) %>% 
  dcast(date + Violent_Homicide + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + t  ~  month , fun.aggregate = length)

df <- df %>% 
  mutate(date = as.Date(date)) %>% 
  mutate(year = year(date)) %>% 
  dcast(date + Violent_Homicide + Drug_Drug + Property_Burglary + Property_Theft + Violent_Assault + Violent_Robbery + t + January+February+March+April+May+June+July+August+September+October+November ~ year , fun.aggregate = length)

df$tint <- as.integer(difftime(df$date,as.Date("2020-01-01"),units="days"))*df$t


acf(df[[category]],lag.max = 20)
pacf(df[[category]],lag.max = 20)

xnames = c(
  "t",
  "tint",
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "2017",
  "2018",
  "2019"
)


category = "Violent_Homicide" 
y <- df$Violent_Homicide
xreg = df[, xnames]
ts_arima_dynamicte <- tsglm(y, model=list(past_obs=1, past_mean=2),xreg=xreg, distr="poisson")

res_coefs[[category]] <- ts_arima_dynamicte$coefficients
res_fittedvalues[[category]] <- ts_arima_dynamicte$fitted.values
res_arimas[[category]] <- ts_arima_dynamicte
pdf("homicide-its-dynamicte.pdf") 
plot(df$date,y, cex = 0.3, xlab='date',ylab='# homicides/week')
lines(df$date,ts_arima_dynamicte$fitted.values,col="black", lwd=1.5)
abline(v = as.Date("2020-01-01"), col="black", lty="dotted")
dev.off() 


texreg(c(extract.tsglmarima(ts_arima),extract.tsglmarima(ts_arima_dynamicte) ))



