import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from Pastillas_Prod_Analysis import *


def plot_variable_importance(variable_name):
  """Plots the importance of a single variable as a horizontal bar plot.

  Args:
    variable_name: The name of the variable to plot.
  """
  importance = average_importance[variable_name]

  plt.figure(figsize=(8, 2))  # Adjust figure size as needed
  plt.barh([variable_name], [importance], color='lightblue')
  plt.xlabel('Importance')
  plt.xlim(0, 1)  # Adjust x-axis limit
  plt.title(f'Importance of {variable_name}')
  plt.text(importance + 0.001, 0, str(round(importance*100, 3))+'%', va='center')  # Add value label
  #plt.show()
  return plt

def plot_partial_dependence(model, X, features):
    # Plot using PartialDependenceDisplay
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=features,
        grid_resolution=50
    )
    #plt.show()
    return plt

def interpretation(variable):
  
  importance = round(average_importance[variable]*100, 2)

  average_impact_pos = round(mean_pos_impacts_avg[variable]*100, 3)
  average_impact_neg = round(mean_neg_impacts_avg[variable]*100, 3)


  importance_inter = ''
  impact_inter = ''
  sugerencia = ''

# Interpretation of importance
  if importance >= 15:
    
    importance_inter += f'Si buscamos entender el porqué la fiabrilidad varía entre los lotes ACTUALES, el {importance}% de las variaciones de friabilidad son explicadas por cambios en {variable}.\n'
    importance_inter += f'Es decir, {variable} puede ser o no causa directa para cambios en friabilidad, pero este indicador sugiere que {variable} está variando mucho entre lotes y muestras.' 
    importance_inter += f' Y eso afectó la variabilidad de la friabilidad.'
    sugerencia += f'Para reducir las altas variaciones en la friabilidad, se sugiere tener un control más estricto sobre {variable}.'

  elif importance >= 10 and importance < 15:

    importance_inter += f'Si buscamos entender el porqué la fiabrilidad varía entre los lotes ACTUALES, el {importance}% de las variaciones de friabilidad son explicadas por cambios en {variable}.\n'
    importance_inter += f'Es decir, {variable} puede ser o no causa directa para cambios en friabilidad, pero este indicador sugiere que {variable} está variando entre lotes y muestras.' 
    importance_inter += f' Y eso afectó la variabilidad de la friabilidad.'
    sugerencia += f'Para reducir las altas variaciones en la friabilidad, tener un control más estricto sobre {variable} pudiera reducir dicha variabilidad.'

  else:
    importance_inter += f'La variable {variable} no suele variar mucho entre los lotes y muestras. Por ende, el modelo encontró una importancia baja de {importance}%.'
    sugerencia += f'Esta variable está bajo control respecto a la variabilidad de la friabilidad. Es decir, con lo que actualmente varía {variable} la friabilidad no varía mucho.'

# Interpretation of impact


  if (average_impact_pos > 2.5 or average_impact_neg < -2.5) and variable != 'Dureza_VM':
    impact_inter += f'La vaiable {variable} tiene un efecto significativo en la friabilidad de un lote. Por cada unidad aumentada en {variable}, la friabilidad aumenta en promedio {average_impact_pos}%.\n'
    impact_inter += f'A su vez, por cada unidad disminuida en {variable}, en promedio la friabilidad disminuye en {average_impact_neg}%'
    sugerencia += f'\nAhora, visto que el impacto de {variable} es pronunciado sobre la friabilidad, se sugirere redcuir el valor de {variable} para reducir la friabilidad.'

  elif variable == 'Dureza_VM': # Only variable with negative correlation
    impact_inter += f'La vaiable {variable} tiene un efecto significativo en la friabilidad de un lote. Por cada unidad aumentada en {variable}, la friabilidad disminuye en promedio {-1*average_impact_pos}%.\n'
    impact_inter += f'A su vez, por cada unidad disminuida en {variable}, en promedio la friabilidad aumenta en {abs(average_impact_neg)}%'
    sugerencia += f'\nAhora, visto que el impacto de {variable} es significativo sobre la friabilidad, se sugirere aumentar el valor de {variable} para reducir la friabilidad.'

  elif average_impact_pos > 1 or average_impact_neg < -1:
    impact_inter += f'La vaiable {variable} tiene un efecto notable en la friabilidad de un lote. Por cada unidad aumentada en {variable}, la friabilidad aumenta en promedio {average_impact_pos}%.\n'
    impact_inter += f'A su vez, por cada unidad disminuida en {variable}, en promedio la friabilidad disminuye en {average_impact_neg}%'
    sugerencia += f'\nAhora, visto que el impacto de {variable} es notable sobre la friabilidad de un lote, reducir el valor de {variable} puede ser útil para reducir la friabilidad.'

  elif average_impact_pos > 0.7 or average_impact_neg < -0.7:
    impact_inter += f'La vaiable {variable} tiene un efecto discreto en la friabilidad de un lote. Por cada unidad aumentada en {variable}, la friabilidad aumenta en promedio {average_impact_pos}%. \n'
    impact_inter += f'A su vez, por cada unidad disminuida en {variable}, en promedio la friabilidad disminuye en {average_impact_neg}%'
    sugerencia += f'\nFinalmente, visto que el impacto de {variable} es poco destacable sobre la friabilidad de un lote, reducir el valor de {variable} no es lo recomendado para reducir la friabilidad.'
  
  else: 
    impact_inter += f'El impacto directo de la variable {variable} sobre la friabilidad no es considerable ({average_impact_pos}% por cada unidad aumentada, y {average_impact_neg}% por cada unidad decrementada).'
    sugerencia += f'\nFinalmente, visto que el impacto de {variable} es poco destacable sobre la friabilidad de un lote, reducir el valor de {variable} no es lo recomendado para reducir la friabilidad.'

  return importance_inter, impact_inter, sugerencia


def variable_analysis(variable):

  importance_int_var, impact_int_var, suggest_var = interpretation(variable)

  st.title(f"Analisis de {variable}")

  
  st.pyplot(plot_variable_importance(variable))

  st.write(f'{importance_int_var}')

  
  st.pyplot(plot_partial_dependence(Random_Forest, x_train_RF, [variable]))
  st.subheader("Interpretation:")

  st.write(f'{impact_int_var}\n')
  st.write(f'{suggest_var}')


st.title("Análisis de variables")

selected_variable = st.selectbox("Variable: ", x.columns)

if st.button("Go!"):
  variable_analysis(selected_variable)
