library(tidyverse)

# update file locations accordingly 
incidents_raw = read.csv(file = 'final_incidents_2021-05-28.csv')


incidents  = incidents_raw %>% unite(category, c("category_1", "category_2")) %>% select(-category_3) %>% group_by(city, category, std_date) %>% summarize (n=n()) %>% ungroup
#categories = incidents%>%select(category)%>%unique
#cities     = incidents%>%select(city)%>%unique
#days       = incidents%>%select(std_date)%>%unique

incident_series = incidents %>% pivot_wider(values_from = n, names_from = std_date) #%>% select(order(colnames(.))) %>% arrange(city)
#write.csv(incident_series, 'incident_series_newyork_update.csv')

write.csv(incident_series, 'incident_series_update.csv')
