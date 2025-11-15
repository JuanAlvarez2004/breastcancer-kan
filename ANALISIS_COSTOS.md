## 1. COSTO DE FALSO NEGATIVO (cost_fn)

### Definición
Un **falso negativo** ocurre cuando el algoritmo de diagnóstico falla en detectar cáncer de mama cuando está realmente presente. Esto resulta en diagnóstico tardío, estadificación avanzada y tratamiento significativamente más costoso.

### Fundamento Teórico

En Colombia, el diagnóstico tardío es un problema crítico de salud pública. Según estudios epidemiológicos recientes:

- **57.5% de los casos nuevos de cáncer de mama se diagnostican en estadios tardíos** (regional o metastásico)
- El estadio al diagnóstico es el factor pronóstico más importante para sobrevida y costo total del tratamiento
- Los tratamientos en estadios avanzados requieren quimioterapia, radioterapia y múltiples hospitalizaciones

**Referencia:**

Consultorsalud. (2017). *57% de casos de cáncer de mama se diagnostican en etapa tardía en Colombia*. Recuperado de https://consultorsalud.com/57-casos-cancer-de-mama-tardio-col/

### Estudio de Costos Directos

El estudio más completo sobre costos directos del cáncer de mama en Colombia fue publicado por Gamboa et al. en la **Revista Colombiana de Cancerología** (indexada en Scielo). Este trabajo analizó costos por estadio de presentación durante 5 años de seguimiento:

| Estadio | Costo 2016 (COP) | Componentes |
|---------|-----------------|------------|
| Tempranos (I-IIA) | $51,934,885 | Cirugía, radioterapia limitada |
| Localmente avanzados (IIIA-IIIC) | $63,912,213 | Cirugía, quimioterapia, radioterapia |
| Metastásico (IV) | $144,400,865 | Quimioterapia prolongada, radioterapia paliativa, hospitalización |

**Fuente:**
> Gamboa, O., Samper, B., & Serrano, A. (2016). Costos directos de la atención del cáncer de mama en Colombia. *Revista Colombiana de Cancerología*, 20(2), 55-66.
> - SciELO: http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0123-90152016000200002
> - DOI: 10.1016/j.rccan.2016.02.003

**Desglose del costo en estadio metastásico (mayor componente):**
- Quimioterapia: 75-88% del costo total
- Cirugía/procedimientos: 5-10%
- Radioterapia: 5-10%
- Hospitalización y manejo de efectos adversos: 5-10%

### Distribución de Diagnósticos Tardíos

El Instituto Nacional de Cancerología de Colombia, en su Boletín Epidemiológico 2017, reporta dos estudios sobre la distribución de estadios:

#### Estudio 1: Mujeres con Cáncer de Mama en Bogotá (Piñeros et al., 2011)

Realizado en pacientes de Bogotá, mostró la siguiente distribución:

- Estadios tempranos (I y IIA): 31.2% de los casos
- Estadios localmente avanzados (IIB, IIIA, IIIB, IIIC): 57.1% de los casos
- Estadio metastásico (IV): 4.5% de los casos
- Total diagnosticado: 92.8%

#### Estudio 2: Anuario Estadístico del Instituto Nacional de Cancerología (2014)

El INC registró la distribución completa de estadios en sus pacientes durante 2014:
- Carcinoma in situ: 6.7%
- Estadios tempranos (I y IIA): 22.8%
- Estadios localmente avanzados (IIB, IIIA, IIIB, IIIC): 47.4%
- Estadio metastásico (IV): 11.9%
- Total reportado: 88.8%

Del total de diagnósticos tardíos reportados en el Anuario 2014 del INC (estudio mas reciente):
```
Diagnósticos tardíos = Localmente Avanzados + Metastásico 
= 47.4% + 11.9% = 59.3% del total
```

**Referencia:**

Instituto Nacional de Cancerología. (2017). *Boletín Epidemiológico 2017*. Recuperado de https://www.cancer.gov.co/recursos_user/files/libros/archivos/B

**Probabilidades**

\[
P(\text{Localmente Avanzado} \mid \text{Tardío}) 
= \frac{47.4}{47.4 + 11.9} 
= \frac{47.4}{59.3} 
= 0.799 \approx 79.9\%
\]

\[
P(\text{Metastásico} \mid \text{Tardío}) 
= \frac{11.9}{47.4 + 11.9} 
= \frac{11.9}{59.3} 
= 0.201 \approx 20.1\%
\]

### Ajuste por Inflación (2016 → 2025)

**Factor de inflación acumulada sector salud:** 

El factor de inflación se calculó con el IPC específico del sector salud desde 2016 hasta septiembre de 2025, utilizando datos del portal "Así Vamos en Salud":

| Período | Inflación Salud | Factor Multiplicativo |
|---------|-----------------|-----------------|
| 2016 | 8.14% | 1.0000 |
| 2017 | 6.34% | 1.0634 |
| 2018 | 4.33% | 1.0433 |
| 2019 | 2.82% | 1.0282 |
| 2020 | 4.96% | 1.0496 |
| 2021 | 3.98% | 1.0398 |
| 2022 | 9.53% | 1.0953 |
| 2023 | 9.49% | 1.0949 |
| 2024 | 5.54% | 1.0554 |
| 2025p | 5.67% | 1.0567 |

**Referencia:**

Así Vamos en Salud. (2025). *Evolución del IPC y del IPC Salud*. Recuperado de https://www.asivamosensalud.org/indicadores/financiamiento/evolucion-del-ipc-y-del-ipc-salud

### Cálculo de precios ajustados a 2025:

### **Factor Inflacionario Acumulado**

\[
F = 1.0634 \times 1.0433 \times 1.0282 \times 1.0496 \times 1.0398 \times 1.0953 \times 1.0949 \times 1.0554 \times 1.0567
\]

\[
F = 1.66507361404
\]

**Costos base en 2016**

| Estadio | Costo 2016 (COP) |
|---------|------------------|
| Tempranos (I–IIA) | 51,934,885 |
| Localmente avanzados (IIIA–IIIC) | 63,912,213 |
| Metastásico (IV) | 144,400,865 |


**Cálculo de los Costos Actualizados a 2025**

La fórmula general es:

\[
\text{Costo}_{2025} = \text{Costo}_{2016} \times F
\]

Donde \( F = 1.66507361404 \)

**Estadio I–IIA (Temprano)**

\[
51,934,885 \times 1.66507361404 = 86,475,406
\]

**Costo actualizado:  
\$\86,475,406 COP**

**Estadio IIIA–IIIC (Regional)**

\[
63,912,213 \times 1.66507361404 = 106,418,539
\]

**Costo actualizado:  
\$\106,418,539 COP**

**Estadio IV (Metastásico)**

\[
144,400,865 \times 1.66507361404 = 240,438,070
\]

**Costo actualizado:  
\$\240,438,070 COP**

**Resultados Finales**

| Estadio | Costo 2016 | Costo Ajustado a 2025 (COP) |
|---------|------------|------------------------------|
| Tempranos (I–IIA) | 51,934,885 | **86,475,406** |
| Localmente avanzados (IIIA–IIIC) | 63,912,213 | **106,418,539** |
| Metastásico (IV) | 144,400,865 | **240,438,070** |

### Cálculo del Costo Ponderado

**Costo promedio de Falso Negativo:**

\[
\text{cost\_fn} = 
P(\text{Localmente Avanzado}) \times \text{Costo}_{LA} \;+\;
P(\text{Metastásico}) \times \text{Costo}_{M}
\]

\[
\text{cost\_fn} =
(0.799 \times 106,418,539) +
(0.201 \times 240,438,070)
\]

\[
\text{cost\_fn} =
85,028,412 \;+\; 48,328,052
\]

\[
\text{cost\_fn} = 133,356,464 \;\text{COP}
\]


### Valor Final - cost_fn

- **En Pesos Colombianos:** $133,356,464 COP
---

## 2. COSTO DE FALSO POSITIVO (cost_fp)

### Definición
Un **falso positivo** ocurre cuando el algoritmo de diagnóstico genera una alarma (resultado positivo o sospechoso) pero la paciente no tiene cáncer de mama. Esto requiere investigación adicional mediante biopsia y seguimiento para descartar malignidad.

### Decisión Metodológica
Para la estimación del costo económico asociado a un falso positivo, se ha optado por utilizar la ruta mas completa para poder detectarlo, que incluye la secuencia diagnóstica ACAF seguida de biopsia Trucut.

#### Justificación
La selección de esta ruta se fundamenta en el principio de precaución económica y representa el escenario del peor caso (worst-case scenario) por las siguientes razones:
1. **Conservadurismo Económico**
Al utilizar el costo más alto posible, se garantiza que:
- No se subestimen los costos reales del sistema de salud
- Las estimaciones económicas sean robustas ante variabilidad clínica
- Se capture el impacto económico máximo de los falsos positivos

2. **Minimización de Riesgo de Subestimación**
Al asumir el peor escenario económico:
- Se evita minimizar el impacto financiero real de los falsos positivos
- Se protege la validez del análisis ante variaciones en protocolos clínicos
- Se considera la heterogeneidad de la práctica médica entre instituciones

### Componentes del Costo de Falso Positivo
**Procedimiento 1: ACAF SENO ($498,000)**

- Biopsia por punción con aguja fina de mama: $310,000
- Ecografía como guía para procedimientos: $120,000
- Estudio anatomopatológico básico: $68,000

**Procedimiento 2: BIOPSIA TRUCUT SENO ($667,000)**
- Biopsia de mama con aguja Trucut: $410,000
- Ecografía como guía para procedimientos: $120,000
- Estudio de coloración básica en biopsia: $137,000

**Referencia:**

Liga Colombiana Contra el Cáncer. (2025). *Tarifas para particulares*. Recuperado de https://www.ligacontraelcancer.com.co/tarifas-para-particulares/ 

### Total Costo de Falso Positivo (2025)

```
cost_fp = $498,000 + $667,000
cost_fp = $1,165,000 COP
```

## 3. COMPARATIVA: COSTO FALSO NEGATIVO vs FALSO POSITIVO

### Resumen de Costos Finales (2025)

Con base en los cálculos detallados en las secciones anteriores, los costos unitarios actualizados a 2025 son:

| Tipo de Error | Costo Unitario (COP) |
|---------------|----------------------|
| **Falso Negativo (FN)** | $133,356,464 |
| **Falso Positivo (FP)** | $1,165,000 |

### Razón de Costos

La relación entre el costo de un falso negativo y un falso positivo es:

\[
\text{Razón} = \frac{\text{cost\_fn}}{\text{cost\_fp}} = \frac{133,356,464}{1,165,000} = 114.46
\]

```
Razón cost_fn / cost_fp = $133,356,464 / $1,165,000 ≈ 114.5:1
```

### Interpretación del Análisis Económico

El costo de un falso negativo (diagnóstico tardío) es aproximadamente **114.5 veces mayor** que el costo de un falso positivo (alarma falsa que requiere estudios adicionales). Esta significativa diferencia tiene importantes implicaciones:

1. **Importancia crítica de la sensibilidad:** El costo económico de no detectar un caso de cáncer (\$133.4 millones COP) es sustancialmente superior al costo de investigar una alarma falsa ($1.2 millones COP)

2. **Impacto económico del diagnóstico tardío:** Los tratamientos en estadios avanzados (quimioterapia prolongada, radioterapia, hospitalizaciones) representan costos exponencialmente mayores que los procedimientos diagnósticos de confirmación

3. **Justificación de alta sensibilidad:** Desde una perspectiva de costo-efectividad, es preferible tolerar múltiples falsos positivos antes que incurrir en un falso negativo. Por ejemplo:
   - 1 Falso Negativo = $133,356,464 COP
   - 114 Falsos Positivos = $132,610,000 COP
   - El sistema de salud puede costear hasta 114 investigaciones innecesarias con el mismo costo de un solo diagnóstico tardío

4. **Beneficio sanitario y económico:** La detección temprana no solo mejora el pronóstico y la sobrevida de las pacientes, sino que también reduce significativamente la carga económica sobre el sistema de salud

### Desglose de Componentes de Costo

**Falso Negativo ($133,356,464 COP):**
- Tratamiento en estadio localmente avanzado (79.9% probabilidad): $106,418,539
- Tratamiento en estadio metastásico (20.1% probabilidad): $240,438,070
- Costo ponderado incluye: quimioterapia extendida, cirugías complejas, radioterapia, hospitalización, manejo de complicaciones y efectos adversos

**Falso Positivo ($1,165,000 COP):**
- ACAF (Aspiración con Aguja Fina): $498,000
- Biopsia Trucut guiada por ecografía: $667,000
- Incluye: procedimientos, estudios anatomopatológicos y seguimiento

### Tabla de Síntesis Final

| Métrica | Valor |
|---------|-------|
| **Costo Falso Negativo (COP)** | $133,356,464 |
| **Costo Falso Positivo (COP)** | $1,165,000 |
| **Razón FN/FP** | 114.5:1 |
| **Factor Inflación Total (2016-2025)** | 1.665 (66.5%) |
| **Período de Validez** | 2025 |
| **Metodología** | Costos directos, worst-case scenario |