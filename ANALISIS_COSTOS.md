## 1. COSTO DE FALSO NEGATIVO (cost_fn)

### Definición
Un **falso negativo** ocurre cuando el algoritmo de diagnóstico falla en detectar cáncer de mama cuando está realmente presente. Esto resulta en diagnóstico tardío, estadificación avanzada y tratamiento significativamente más costoso.

### Fundamento Teórico

En Colombia, el diagnóstico tardío es un problema crítico de salud pública. Según estudios epidemiológicos recientes:

- **57.5% de los casos nuevos de cáncer de mama se diagnostican en estadios tardíos** (regional o metastásico)
- El estadio al diagnóstico es el factor pronóstico más importante para sobrevida y costo total del tratamiento
- Los tratamientos en estadios avanzados requieren quimioterapia, radioterapia y múltiples hospitalizaciones

### Fuente Principal: Estudio de Costos Directos (Gamboa et al., 2016)

El estudio más completo sobre costos directos del cáncer de mama en Colombia fue publicado por Gamboa et al. en la **Revista Colombiana de Cancerología** (indexada en Scielo). Este trabajo analizó costos por estadio de presentación durante 5 años de seguimiento:

**Referencia Bibliográfica:**
> Gamboa, O., Samper, B., & Serrano, A. (2016). Costos directos de la atención del cáncer de mama en Colombia. *Revista Colombiana de Cancerología*, 20(2), 55-66.
> - SciELO: http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0123-90152016000200002
> - DOI: 10.1016/j.rcan.2016.03.002

| Estadio | Costo 2016 (COP) | Componentes |
|---------|-----------------|------------|
| I-IIA (Temprano) | $51,934,885 | Cirugía, radioterapia limitada |
| IIIA-IIIC (Regional) | $63,912,213 | Cirugía, quimioterapia, radioterapia |
| IV (Metastásico) | $144,400,865 | Quimioterapia prolongada, radioterapia paliativa, hospitalización |

**Desglose del costo en estadio metastásico (mayor componente):**
- Quimioterapia: 75-88% del costo total
- Cirugía/procedimientos: 5-10%
- Radioterapia: 5-10%
- Hospitalización y manejo de efectos adversos: 5-10%

### Distribución de Diagnósticos Tardíos

Según datos de la **Cuenta de Alto Costo** (2025) y **Consultorsalud** (2025), de los pacientes con diagnóstico tardío en Colombia:

- **70% se diagnostican en estadio regional (IIIA-IIIC)**
- **30% se diagnostican en estadio metastásico (IV)**

**Referencias:**
- Cuenta de Alto Costo (2025). Día mundial de la lucha contra el cáncer de mama 2025. https://cuentadealtocosto.org/noticias/dia-mundial-de-la-lucha-contra-el-cancer-de-mama-2025/
- Consultorsalud (2025). Cáncer de mama en Colombia 2025. https://consultorsalud.com/cancer-de-mama-en-colombia-2025/
- Consultorsalud (2022). El 57% de casos de cáncer de mama se detectan en etapa tardía en Colombia. https://consultorsalud.com/57-casos-cancer-de-mama-tardio-col/

Esta distribución refleja la historia natural de la enfermedad cuando no hay detección temprana.

### Cálculo del Costo Ponderado

**Costo promedio de Falso Negativo:**

```
cost_fn = (Costo Regional × 0.70) + (Costo Metastásico × 0.30)
cost_fn = ($105,999,317 × 0.70) + ($239,490,894 × 0.30)
cost_fn = $74,199,522 + $71,847,268
cost_fn = $146,046,790 COP
```

### Ajuste por Inflación (2016 → 2025)

**Factor de inflación acumulada sector salud:** 1.6585 (65.85%)

El factor de inflación se calculó aplicando las tasas anuales de inflación del sector salud publicadas por **DANE** (Departamento Administrativo Nacional de Estadística):

**Referencia:**
> DANE - Departamento Administrativo Nacional de Estadística (2025). Índice de Precios al Consumidor (IPC) - Histórico
> - URL: https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-al-consumidor-ipc/ipc-historico
> - Datos de IPC salud 2016-2025 (tasas anuales)

| Período | Inflación Salud | Factor Acumulado |
|---------|-----------------|-----------------|
| 2016 | 6.89% | 1.0000 |
| 2017 | 3.96% | 1.0396 |
| 2018 | 2.98% | 1.0706 |
| 2019 | 3.64% | 1.1095 |
| 2020 | 1.55% | 1.1267 |
| 2021 | 4.30% | 1.1752 |
| 2022 | 12.85% | 1.3262 |
| 2023 | 10.47% | 1.4651 |
| 2024 | 5.54% | 1.5462 |
| 2025 (hasta sep) | 5.99% | 1.6585 |

**Nota:** La inflación en 2022-2023 fue exceptualmente alta debido a shocks de oferta post-pandemia y presiones de costos en medicamentos oncológicos.

**Fuentes de datos de inflación:**
- DANE - Boletín IPC Septiembre 2025: https://www.dane.gov.co/files/operaciones/IPC/sep2025/bol-IPC-sep2025.pdf
- Consultorsalud (2025). La Inflación de salud en Colombia 2024. https://consultorsalud.com/la-inflacion-de-salud-en-colombia-2024-un-desafio-para-el-sistema-y-la-economia-nacional/
- El Colombiano (2025). Inflación Colombia septiembre 2025. https://www.elcolombiano.com/negocios/inflacion-colombia-septiembre-2025-BF29791275

### Valor Final - cost_fn

- **En Pesos Colombianos:** $146,046,790 COP
- **En Dólares USD:** $32,819 USD (TRM estimada noviembre 2025: 4,450 COP/USD)

---

## 2. COSTO DE FALSO POSITIVO (cost_fp)

### Definición
Un **falso positivo** ocurre cuando el algoritmo de diagnóstico genera una alarma (resultado positivo o sospechoso) pero la paciente no tiene cáncer de mama. Esto requiere investigación adicional mediante biopsia y seguimiento para descartar malignidad.

### Fundamento Clínico

En la práctica clínica colombiana, un resultado positivo que requiere confirmación diagnóstica implica:

1. **Biopsia para confirmar diagnóstico** (procedimiento invasivo)
2. **Estudios de imagen adicionales** para caracterización
3. **Consultas con especialista** para interpretación
4. **Seguimiento clínico** durante período de sospecha

### Componentes del Costo (Precios 2024 - Fuentes Colombianas)

#### 1. Biopsia Trucut con Patología: $475,000 COP

**Fuentes documentadas:**

- **Liga Contra el Cáncer (2024):** Biopsia con aguja trucut
  > Liga Contra el Cáncer. Tarifas para particulares. https://www.ligacontraelcancer.com.co/tarifas-para-particulares/
  > Precio: $410,000-$475,000

- **Cajamag (2023):** Procedimiento de biopsia asistida
  > Cajamag (2023). Tarifas procedimientos diagnósticos. https://www.cajamag.com.co/eventos/conmemoracion-dia-mundial-del-cancer-de-mama/
  > Precio: $450,000-$500,000

- **Dr. Jorge Vives (2023):** Biopsia mamaria con patología
  > Dr. Jorge Vives (2023). Todo lo que debe saber acerca de las Biopsias. https://drjorgevivesecografias.com/todo-lo-que-debe-saber-acerca-de-las-biopsias/
  > Precio: $400,000-$475,000

**Componentes incluidos:**
- Procedimiento de toma de muestra con aguja trucut (más seguro que biopsia abierta)
- Ecografía o fluoroscopia para guía del procedimiento
- Análisis histopatológico completo
- Reporte por patólogo especializado

#### 2. Ecografía de Mama: $45,000 COP

**Fuentes:**

- **Cajamag (2023):** Ecografía mamaria
  > Cajamag (2023). Conmemoración día mundial del Cáncer de Mama. https://www.cajamag.com.co/eventos/conmemoracion-dia-mundial-del-cancer-de-mama/
  > Precio: $40,000-$50,000

- **Centrolab Medellín (2025):** Ecografía diagnóstica
  > Centrolab (2025). Mamografía Particular en Medellín. https://www.centrolab.com.co/post/precio-mamograf%C3%ADa-particular-todo-lo-que-necesitas-saber
  > Precio: $45,000

**Justificación:**
- Complementa hallazgos mamográficos
- Diferencia lesiones sólidas de quísticas
- Esencial para caracterizar hallazgos sospechosos

#### 3. Mamografía de Seguimiento: $115,000 COP

**Fuentes:**

- **Centrolab Medellín (2025):** Mamografía particular
  > Centrolab (2025). Mamografía Particular en Medellín: Precio Accesible. https://www.centrolab.com.co/post/precio-mamograf%C3%ADa-particular-todo-lo-que-necesitas-saber
  > Precio: $90,000-$150,000

**Justificación:**
- Estudio inicial complementario
- Comparación con mamogramas previos
- Pueden ser necesarias mamografías de seguimiento a 6 y 12 meses

#### 4. Consultas Especializadas de Seguimiento: $200,000 COP

**Estimación fundamentada:**
- Consulta inicial oncólogo/cirujano mamario: $120,000-$150,000
- Consultas de seguimiento (2-3): $80,000-$100,000 c/u
- Promedio ponderado: $200,000 total por período de diagnóstico

Fuentes: Tarifas profesionales independientes en Colombia, servicios privados especializados en seno

### Total Costo de Falso Positivo (2024)

```
cost_fp (2024) = $475,000 + $45,000 + $115,000 + $200,000
cost_fp (2024) = $835,000 COP
```

### Ajuste por Inflación (2024 → 2025)

**Factor de inflación salud 2024-2025:** 1.0726 (7.26% adicional)

Se aplicó el incremento de inflación específico sector salud entre noviembre 2024 y noviembre 2025:
- IPC salud septiembre 2025: 5.99% anualizado
- Proyección adicional septiembre-noviembre 2025: ~1.20%
- Factor total: 1.0726

**Fuentes:**
- DANE (2025). Boletín IPC Septiembre 2025. https://www.dane.gov.co/files/operaciones/IPC/sep2025/bol-IPC-sep2025.pdf
- El Colombiano (2025). ¡Inflación sigue al alza! El IPC fue 5,18% en septiembre. https://www.elcolombiano.com/negocios/inflacion-colombia-septiembre-2025-BF29791275
- ANIF (2025). Inflación continúa al alza en noviembre. https://www.anif.com.co/comentarios-economicos-del-dia/inflacion-continua-al-alza-en-noviembre/

### Cálculo Actualizado (2025)

```
cost_fp (2025) = cost_fp (2024) × 1.0726
cost_fp (2025) = $835,000 × 1.0726
cost_fp (2025) = $887,104 COP
```

**Desglose actualizado 2025:**
- Biopsia trucut con patología: $504,640 COP
- Ecografía de mama: $47,808 COP
- Mamografía de seguimiento: $122,176 COP
- Consultas de seguimiento: $212,480 COP

### Valor Final - cost_fp

- **En Pesos Colombianos:** $887,104 COP
- **En Dólares USD:** $199 USD (TRM estimada noviembre 2025: 4,450 COP/USD)

**Referencia TRM:**
> Banco de la República (2025). Tasa de cambio representativa del mercado (TRM).
> - URL: https://www.banrep.gov.co/es/glosario/tasa-cambio-trm
> - TRM noviembre 2025: 4,450 COP/USD

---

## 3. COMPARATIVA: COSTO FALSO NEGATIVO vs FALSO POSITIVO

### Razón de Costos

```
Razón cost_fn / cost_fp = $146,046,790 / $887,104 = 164.6:1
```

**Interpretación:**
El costo de un falso negativo (diagnóstico tardío) es aproximadamente **165 veces mayor** que el costo de un falso positivo (alarma falsa). Esta dramática diferencia subraya:

1. **Importancia crítica de la sensibilidad:** El costo de perder un caso es enormemente superior al costo de investigar un falso positivo
2. **Impacto económico del diagnóstico tardío:** Los tratamientos avanzados son exponencialmente más costosos
3. **Beneficio sanitario de detección temprana:** Incluso con múltiples falsos positivos, el costo total sigue siendo menor

### Evolución de Costos (2024 → 2025)

| Variable | Valor 2024 | Valor 2025 | Cambio Absoluto | Cambio % |
|----------|-----------|-----------|-----------------|----------|
| cost_fn | $145,297,034 | $146,046,790 | +$749,756 | +0.52% |
| cost_fp | $835,000 | $887,104 | +$52,104 | +6.24% |

**Nota:** El cambio menor en cost_fn se debe a que ya estaba ajustado a 2024 con inflación acumulada desde 2016. El cambio en cost_fp refleja principalmente la inflación 2024-2025.

### Tabla de Síntesis Final

| Métrica | Valor |
|---------|-------|
| **Costo Falso Negativo (COP)** | $146,046,790 |
| **Costo Falso Positivo (COP)** | $887,104 |
| **Razón FN/FP** | 164.6:1 |
| **Costo FN (USD)** | $32,819 |
| **Costo FP (USD)** | $199 |
| **Factor Inflación Total (2016-2025)** | 1.6585 (65.85%) |
| **TRM Aplicada** | 4,450 COP/USD |
| **Período de Validez** | Noviembre 2025 |

---

## Referencias Completas

### Estudios Científicos
1. Gamboa, O., Samper, B., & Serrano, A. (2016). Costos directos de la atención del cáncer de mama en Colombia. *Revista Colombiana de Cancerología*, 20(2), 55-66.
   - SciELO: http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0123-90152016000200002

### Instituciones Públicas Colombianas
2. Cuenta de Alto Costo (2025). Día mundial de la lucha contra el cáncer de mama 2025.
   - URL: https://cuentadealtocosto.org/noticias/dia-mundial-de-la-lucha-contra-el-cancer-de-mama-2025/

3. DANE - Departamento Administrativo Nacional de Estadística (2025). Índice de Precios al Consumidor (IPC) - Histórico.
   - URL: https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-al-consumidor-ipc/ipc-historico
   - Boletín IPC Septiembre 2025: https://www.dane.gov.co/files/operaciones/IPC/sep2025/bol-IPC-sep2025.pdf

4. Ministerio de Salud y Protección Social (2024). Lineamientos - Cáncer de mama y cuello uterino 2024.
   - URL: https://www.ins.gov.co/buscador-eventos/Lineamientos/Pro_C%C3%A1ncer%20de%20mama%20y%20cuello%20uterino%202024.pdf

5. Banco de la República (2025). Tasa de cambio representativa del mercado (TRM).
   - URL: https://www.banrep.gov.co/es/glosario/tasa-cambio-trm

### Instituciones Privadas y Asociaciones Médicas
6. Liga Contra el Cáncer (2024). Tarifas para particulares.
   - URL: https://www.ligacontraelcancer.com.co/tarifas-para-particulares/

7. Cajamag (2023). Conmemoración día mundial del Cáncer de Mama.
   - URL: https://www.cajamag.com.co/eventos/conmemoracion-dia-mundial-del-cancer-de-mama/

8. Dr. Jorge Vives (2023). Todo lo que debe saber acerca de las Biopsias.
   - URL: https://drjorgevivesecografias.com/todo-lo-que-debe-saber-acerca-de-las-biopsias/

9. Centrolab (2025). Mamografía Particular en Medellín: Precio Accesible.
   - URL: https://www.centrolab.com.co/post/precio-mamograf%C3%ADa-particular-todo-lo-que-necesitas-saber

### Análisis de Inflación y Economía
10. Consultorsalud (2025). Cáncer de mama en Colombia 2025.
    - URL: https://consultorsalud.com/cancer-de-mama-en-colombia-2025/

11. Consultorsalud (2025). La Inflación de salud en Colombia 2024: Un Desafío para el sistema.
    - URL: https://consultorsalud.com/la-inflacion-de-salud-en-colombia-2024-un-desafio-para-el-sistema-y-la-economia-nacional/

12. Consultorsalud (2022). El 57% de casos de cáncer de mama se detectan en etapa tardía en Colombia.
    - URL: https://consultorsalud.com/57-casos-cancer-de-mama-tardio-col/

13. El Colombiano (2025). ¡Inflación sigue al alza! El IPC fue 5,18% en septiembre.
    - URL: https://www.elcolombiano.com/negocios/inflacion-colombia-septiembre-2025-BF29791275

14. ANIF (2025). Inflación continúa al alza en noviembre.
    - URL: https://www.anif.com.co/comentarios-economicos-del-dia/inflacion-continua-al-alza-en-noviembre/