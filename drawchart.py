# importing pandas library 
import pandas as pd 
# import matplotlib library 
import matplotlib.pyplot as plt 
  
# creating dataframe 
df = pd.DataFrame({ 
    'Model': ['Houman and etc', 'Li  and etc', 'Chi and etc ', 'This paper '], 
    'Accuracy': [85, 99, 99.9, 94], 
    'Precision': [73.3, 92, 99, 96],
    'Recall': [74, 92, 99, 90],
    'F1 Score': [72, 92, 99, 93] 
}) 
  
# plotting graph 
df.plot(x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score"], kind="bar")




plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.xticks(rotation=90)
plt.xticks(rotation='horizontal')
plt.legend(loc="lower left", ncol=len(df.columns), fontsize=15)
plt.show()