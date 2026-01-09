# Predicting airplane Range in km

## 🧾 Project haqida

- barcha featurelarga kura Airplane ni Range_(km) yani bir zapravqada nechi kilometer yurishini predict qiladigon project 

* **Type:** Regression
* **Target:** `Range_(km)`


## Raw dataset
        #   Column                       Non-Null Count  Dtype  
        ---  ------                       --------------  -----  
        0   Model                        12377 non-null  object 
        1   Year_of_Manufacture          12377 non-null  int64  
        2   Number_of_Engines            12377 non-null  int64  
        3   Engine_Type                  12377 non-null  object 
        4   Capacity                     12377 non-null  int64  
        5   Range_(km)                   12377 non-null  int64  
        6   Fuel_Consumption_(L/hour)    12377 non-null  float64
        7   Hourly_Maintenance_Cost_($)  12377 non-null  float64
        8   Age                          12377 non-null  int64  
        9   Sales_Region                 12377 non-null  object 
        10  Price_($)                    12377 non-null  float64

## Final dataset
        #   Column                       Non-Null Count  Dtype  
        ---  ------                       --------------  -----  
        0   Model                        12377 non-null  float64
        1   Year_of_Manufacture          12377 non-null  float64
        2   Number_of_Engines            12377 non-null  float64
        3   Capacity                     12377 non-null  float64
        4   Range_(km)                   12377 non-null  int64  
        5   Fuel_Consumption_(L/hour)    12377 non-null  float64
        6   Hourly_Maintenance_Cost_($)  12377 non-null  float64
        7   Age                          12377 non-null  float64
        8   Sales_Region                 12377 non-null  float64
        9   Price_($)                    12377 non-null  float64
        10  HMC_per_person               12377 non-null  float64
        11  Engine_Power_Factor          12377 non-null  float64
        12  Price_per_Seat               12377 non-null  float64
        13  Seats_per_Engine             12377 non-null  float64
        14  Company_Popularity           12377 non-null  float64
        15  FuelCost_Maint_Index         12377 non-null  float64
        16  Engine_to_Capacity           12377 non-null  float64
        17  Engine_Type_Piston           12377 non-null  float64
        18  Engine_Type_Turbofan         12377 non-null  float64
        19  Company_Airbus               12377 non-null  float64
        20  Company_Boeing               12377 non-null  float64
        21  Company_Bombardier           12377 non-null  float64
        22  Company_Cessna               12377 non-null  float64
        23  Age_Group_0-10               12377 non-null  float64
        24  Age_Group_10-20              12377 non-null  float64
        25  Age_Group_20-30              12377 non-null  float64
        26  Age_Group_30-40              12377 non-null  float64
        27  Age_Group_40+                12377 non-null  float64

---
## 🚀 Final Model Training
                           LinearRegression Results                            
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Algorithm        ┃ R2 score ┃ Mean Absolute Error ┃ K-Fold mean ┃ K-Fold std ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ LinearRegression │ 1.0      │ 0.00                │ 1.00        │ 0.00       │
└──────────────────┴──────────┴─────────────────────┴─────────────┴────────────┘ 
kurishimiz mumkinki loyihada eng yaxshi model LinearRegression bulib chiqdi
---
