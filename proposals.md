# Propostas de Trabalho

<img src="https://www.researchgate.net/profile/Jean-Pierre-Raulin/publication/224263574/figure/fig1/AS:302875856130048@1449222639469/Examples-of-VLF-propagation-paths-from-transmitters-triangles-NAA-NDK-NPM-and-NWC-to_W640.jpg" width=300, align="right"> 

## 1. **Identificação de Explosões Solares a partir de sinais VLF com Redes Neurais Profundas**

<br>

<br>

<br>

<br>


*Sinais de VLF podem ser empregados para a detecção de explosões solares [1]. Os sinais de VLF, entretanto, são afetados por uma série de fatores, como período diurno ou noturno 
dos sinais [2], fase do ciclo solar [3], variações no campo magnético [4], além do clima, condições atmosféricas etc. Por outro lado, redes profundas são conhecidas pela capacidade de lidar com grandes volumes de dados complexos e mais recentemente vêm sendo aplicados à pesquisa de explosões solares [5] e mesmo ao tratamento direto de sinais VLF [6] [7] [8]. Este estudo se propõe a avaliar o uso de modelos de redes neurais profundas [9] para identificar explosões solares a partir de dados de VLF coletados na SAVNET [10].*

**Dúvidas e Pontos Relevantes**

* Dados. Modelo 1: Preditores [VLF SAVENET], Objetivo [Explosões Solares]; Modelo 2: Preditores[VLF de outras redes], Objetivo [Explosões Solares]; Modelo 3: Preditores [VLF + Dados como fase solar +/ou Período do Dia], Objetivo [Explosões Solares]

* Alguma métrica das explosões solares, ou apenas identificar sua ocorrência?

* Empregar dados de fase, amplitude ou ambos?

* Potenciais modelos de rede: LSTM, convolucionais, modelos para detecção de anomalia.

* Relevância. Identificação em tempo real seria *i.* viável e *ii.* relevante neste caso?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?

* \*Emprego de modelo multimodal com imagens?

* \*Medidas do passado podem ter de ser corrigidas por uso corrente quando consideramos, por exemplo, variações no campo magnético da terra? [Pavón-Carrasco, F.J., Marsal, S., Campuzano, S.A., & Torta, J.M. (2021). Signs of a new geomagnetic jerk between 2019 and 2020 from Swarm and observatory data. Earth, Planets and Space, 73, 1-11.]. Isso teria algum valor? 

**Potenciais Referências**

[1] Raulin, J.P. (2011). The South America VLF Network (SAVNET): Providing new ground-based diagnostics of space weather conditions. 2011 XXXth URSI General Assembly and Scientific Symposium, 1-4.

[2] Crombie, D.D. (1964). Periodic fading of VLF signals received over long paths during sunrise and sunset. Journal of Research of the National Bureau of Standards, Section D: Radio Science, 27.

[3] Pacini, A.A., & Raulin, J.P. (2006). Solar X-ray flares and ionospheric sudden phase anomalies relationship : A solar cycle phase dependence. Journal of Geophysical Research, 111.

[4] Magalhaes, A.G., Guerche, G., & Raulin, J.P. (2019). Ionosphere D-layer lowering in the region of the South Atlantic Magnetic Anomaly. Journal of Atmospheric and Solar-Terrestrial Physics, 196, 105146.

[5] Nagem, T.A., Qahwaji, R., Ipson, S.S., Wang, Z., & Al-Waisy, A.S. (2018). Deep Learning Technology for Predicting Solar Flares from (Geostationary Operational Environmental Satellite) Data. International Journal of Advanced Computer Science and Applications, 9.

[6] Gross, N., & Cohen, M.B. (2020). VLF Remote Sensing of the D Region Ionosphere Using Neural Networks. Journal of Geophysical Research, 125.

[7] Curro, J., Raquet, J.F., & Borghetti, B.J. (2018). Navigation using VLF signals with artificial neural networks. NAVIGATION.

[8] Wang, J., Huang, Q., Ma, Q., Chang, S., He, J., Wang, H., Zhou, X., Xiao, F., & Gao, C. (2020). Classification of VLF/LF Lightning Signals Using Sensors and Deep Learning Methods. Sensors (Basel, Switzerland), 20.

[9] Goodfellow, I., Bengio, Y., & Courville, A (2016). Deep Learning. MIT Press. Also available online: http://www.deeplearningbook.org.

[10] Raulin, J.P., David, P., Hadano, R., Saraiva, A.C., Correia, E., & Kaufmann, P. (2009). The south America VLF NETwork (SAVNET): Development, installation status, first results. Geofisica Internacional, 48, 253-261.
<br>

<br>

<img src="https://github.com/Rogerio-mack/craam/blob/main/figures/Lambek1980Nature.png?raw=true" width=300, align="right"> 

## 2. **Um Comparativo da Predição do LOD (Lenght Of the Day) com Redes LSTM e outros Métodos**

<br>

<br>

<br>

<br>


*Uma série de fatores influenciam a duração do dia (LOD) ou o tempo de rotação da terra como desde os fluidos dentro do planeta, a redistribuição de massa e os movimentos que ocorrem na atmosfera e nos oceanos, como marés, glaciais e terremotos [1]. Variações nesse tempo são imperceptíveis na superfície da terra e para nosso dia-a-dia, mas causam mudanças nos parâmetros de orientação da Terra (EOP) podendo, assim, afetar de modo significativo medidas de geolocalização (GPS) [2]. Sua medida pode ser feita a partir de técnicas de interferometria de linha de base muito longa (VLBI), como um indicador dos erros nos parâmetros EOP [3], sendo de interesse a predição de seus valores para janelas da ordem de dias ou meses [2]. Diversos métodos regressivos e inteligência artificial têm sido empregados para a predição do LOD [2] [4] [5] e comparados [2]. Liao et. al. [5] emprega um modelo neural simples MLP de 3 camadas para predição, mas não há, até o momento, um comparativo com a predição com modelos mais recentes de redes neurais profundas que podem incorporar o tempo (LSTM), recorrência ou convoluções e podem ser adequados para análise de dados complexos e multimodais [6]. Este estudo se propõe a avaliar o uso de diferentes modelos de redes neurais profundas para predição do LOD comparando-os com os modelos atuais de predição. Também buscará estimar o tempo dos ciclos de LOD (ciclos de 27 dias e 22 anos foram encontrados nos dados LOD [2]).*

**Dúvidas e Pontos Relevantes**

* Os ciclos de 27 dias e 22 anos que foram encontrados nos dados LOD [2] apresentam alguma relação com algum fenômeno conhecido que podemos também explorar?

* No caso de manchas solares, os ciclos de 11 anos foram identificados, e nos dados do campo magnético polar do Sol ciclos de 22 anos. Esses ciclos foram identificados nos dados de entrada? 

* Parece fazer sentido um modelo que empregue recorrência uma vez que temos, por exemplo, um fator inercial de rotação da terra.

* Existem estudos, por exemplo de análise espectral que confirmam esses ciclos observados?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?

* \*Medidas do passado podem ter de ser corrigidas por uso corrente quando consideramos, por exemplo, variações no campo magnético da terra? [Pavón-Carrasco, F.J., Marsal, S., Campuzano, S.A., & Torta, J.M. (2021). Signs of a new geomagnetic jerk between 2019 and 2020 from Swarm and observatory data. Earth, Planets and Space, 73, 1-11.]. Isso teria algum valor? 

**Potenciais Referências**

[1] Lambeck, K. (1980). Changes in length-of-day and atmospheric circulation. Nature, 286(5769), 104–105. doi:10.1038/286104a0.

[2] Menezes G.O.,Raulin, J.P., Ramirez, R.F.H., Silva, L.A., Pamboukian, S.V.D., Merkowitz, S. (2022) Forecasting of Space Geodesy Data and Investigation of the Relationship with the Solar Activity. [*to appear*](https://github.com/Rogerio-mack/craam/blob/main/articles/article_Guilherme_projeto_Novo.pdf)

[3] Malkin, Z. (2009). On comparison of the Earth orientation parameters obtained from different VLBI networks and observing programs. Journal of Geodesy, 83, 547-556.

[4] Modiri, S., Belda, S., Hoseini, M., Heinkelmann, R., Ferrándiz, J.M., & Schuh, H. (2020). Um novo método híbrido para melhorar a previsão de ultracurto prazo de LOD. Journal of Geodesy, 94

[5] Liao, D., Wang, Q.J., Zhou, Y., Liao, X., & Huang, C. (2012). Long-term prediction of the Earth Orientation Parameters by the artificial neural network technique. Journal of Geodynamics, 62, 87-92.

[6] Goodfellow, I., Bengio, Y., & Courville, A (2016). Deep Learning. MIT Press. Also available online: http://www.deeplearningbook.org.


