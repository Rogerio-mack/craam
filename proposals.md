<a name="top"></a>
# Potenciais Propostas de Trabalho

Seguem aqui algumas propostas iniciais de trabalho com dados do CRAAM para discusão.

1. [Identificação de Explosões Solares a partir de sinais VLF com Redes Neurais Profundas](#p1)

2. [Um Comparativo da Predição do LOD (Lenght Of the Day) com Redes LSTM e outros Métodos](#p2)

3. [Análises do Campo Elétrico Atmosférico](#p3)

4. [Análise das Variações de Fase nas Ondas de VLF nos Períodos que Precedem Terromotos](#p4)

5. [Exploração de dados de Ondas Sub-milimétricas em  Fenômenos Astrofísicos](#p5)

<br>

<br>

---

<br>

<a name="p1"></a>
<img src="https://www.researchgate.net/profile/Jean-Pierre-Raulin/publication/224263574/figure/fig1/AS:302875856130048@1449222639469/Examples-of-VLF-propagation-paths-from-transmitters-triangles-NAA-NDK-NPM-and-NWC-to_W640.jpg" width=300, align="right"> 

## 1. **Identificação de Explosões Solares a partir de sinais VLF com Redes Neurais Profundas**

[Back to the top](#top)
<br>

<br>

<br>

<br>


*Sinais de VLF podem ser empregados para a detecção de explosões solares [1]. Os sinais de VLF, entretanto, são afetados por uma série de fatores, como período diurno ou noturno dos sinais [2][11], fase do ciclo solar [3], variações no campo magnético [4], além do clima, condições atmosféricas etc. Por outro lado, redes profundas são conhecidas pela capacidade de lidar com grandes volumes de dados complexos e mais recentemente vêm sendo aplicados à pesquisa de explosões solares [5] e mesmo ao tratamento direto de sinais VLF [6] [7] [8]. Este estudo se propõe a avaliar o uso de modelos de redes neurais profundas [9] para identificar explosões solares a partir de dados de VLF coletados na SAVNET [10].*

### **Referências**

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

[11] Macoleta, E.L., (2015) Contribuição ao estudo de distúrbios ionosféricos utilizando a técnica de VLF. Dissertação (Programa de Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso em: http://tede.mackenzie.br/jspui/handle/tede/1301

### **Fontes de Dados**

[1] SAVNET: (?)

[2] NOAA: https://www.ngdc.noaa.gov/stp/spaceweather.html

[3] AERONET:  https://aeronet.gsfc.nasa.gov/


### **Dúvidas e Pontos Relevantes**

* Modelos diferentes podem ser explorados empregando diferentes variáveis preditoras, desde VLF SAVENET, VLF de outras redes, Dados como fase solar +/ou Período do Dia.

* O modelo pode buscar alguma métrica das explosões solares (por exemplo as classes A, B, C, M, X) ou apenas identificar sua ocorrência.

* Existem outras fontes/redes de dados VLF disponíveis?

* Empregar dados de fase, amplitude ou ambos? Dados de fase parecem ser mais promissores.

* Potenciais modelos de rede: LSTM, convolucionais, modelos para detecção de anomalia.

* Relevância. Identificação em tempo 'real' seria *i.* viável e *ii.* relevante neste caso?

* Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?

* \*Emprego de modelo multimodal com imagens?

* \*Medidas do passado podem ter de ser corrigidas por uso corrente quando consideramos, por exemplo, variações no campo magnético da terra? [Pavón-Carrasco, F.J., Marsal, S., Campuzano, S.A., & Torta, J.M. (2021). Signs of a new geomagnetic jerk between 2019 and 2020 from Swarm and observatory data. Earth, Planets and Space, 73, 1-11.]. Isso teria algum valor? 


<br>

<br>

<a name="p2"></a>
<img src="https://github.com/Rogerio-mack/craam/blob/main/figures/Lambek1980Nature.png?raw=true" width=300, align="right"> 

## 2. **Um Comparativo da Predição do LOD (Lenght Of the Day) com Redes LSTM e outros Métodos**

[Back to the top](#top)
<br>

<br>

<br>

<br>


*Uma série de fatores influenciam a duração do dia (LOD) ou o tempo de rotação da terra como desde os fluidos dentro do planeta, a redistribuição de massa e os movimentos que ocorrem na atmosfera e nos oceanos, como marés, glaciais e terremotos [1]. Variações nesse tempo são imperceptíveis na superfície da terra e para nosso dia-a-dia, mas causam mudanças nos parâmetros de orientação da Terra (EOP) podendo, assim, afetar de modo significativo medidas de geolocalização (GPS) [2]. Sua medida pode ser feita a partir de técnicas de interferometria de linha de base muito longa (VLBI), como um indicador dos erros nos parâmetros EOP [3], sendo de interesse a predição de seus valores para janelas da ordem de dias ou meses [2]. Diversos métodos regressivos e inteligência artificial têm sido empregados para a predição do LOD [2] [4] [5] e comparados [2]. Liao et. al. [5] emprega um modelo neural simples MLP de 3 camadas para predição, mas não há, até o momento, um comparativo com a predição com modelos mais recentes de redes neurais profundas que podem incorporar o tempo (LSTM), recorrência ou convoluções e podem ser adequados para análise de dados complexos e multimodais [6]. Este estudo se propõe a avaliar o uso de diferentes modelos de redes neurais profundas para predição do LOD comparando-os com os modelos atuais de predição. Também buscará estimar o tempo dos ciclos de LOD (ciclos de 27 dias e 22 anos foram encontrados nos dados LOD [2]).*

### **Referências**

[1] Lambeck, K. (1980). Changes in length-of-day and atmospheric circulation. Nature, 286(5769), 104–105. doi:10.1038/286104a0.

[2] Menezes G.O.,Raulin, J.P., Ramirez, R.F.H., Silva, L.A., Pamboukian, S.V.D., Merkowitz, S. (2022) Forecasting of Space Geodesy Data and Investigation of the Relationship with the Solar Activity. [*to appear*](https://github.com/Rogerio-mack/craam/blob/main/articles/article_Guilherme_projeto_Novo.pdf)

[3] Malkin, Z. (2009). On comparison of the Earth orientation parameters obtained from different VLBI networks and observing programs. Journal of Geodesy, 83, 547-556.

[4] Modiri, S., Belda, S., Hoseini, M., Heinkelmann, R., Ferrándiz, J.M., & Schuh, H. (2020). Um novo método híbrido para melhorar a previsão de ultracurto prazo de LOD. Journal of Geodesy, 94

[5] Liao, D., Wang, Q.J., Zhou, Y., Liao, X., & Huang, C. (2012). Long-term prediction of the Earth Orientation Parameters by the artificial neural network technique. Journal of Geodynamics, 62, 87-92.

[6] Goodfellow, I., Bengio, Y., & Courville, A (2016). Deep Learning. MIT Press. Also available online: http://www.deeplearningbook.org.

### **Fontes de Dados**

[1] SAVNET:  (?)

[2] International Earth Rotation and Reference Systems Service (IERS): https://www.iers.org/IERS/EN/Home/home_node.html

[3] Sunspot index and Long-term Solar Observations: https://wwwbis.sidc.be/silso/

[4] The Wilcox Solar Observatory: http://wso.stanford.edu/

[5] Data from [2]: (non-relational database available?)

### **Dúvidas e Pontos Relevantes**

* Os ciclos de 27 dias e 22 anos que foram encontrados nos dados LOD [2] apresentam alguma relação com que fenômenos conhecidos que podemos explorar?

* No caso de manchas solares, os ciclos de 11 anos foram identificados, e nos dados do campo magnético polar do Sol ciclos de 22 anos. Não é claro se esses ciclos foram identificados nos dados de entrada ou têm outra fonte. Existem estudos, por exemplo de análise espectral que confirmam esses ciclos observados?

* Parece fazer sentido um modelo que empregue recorrência uma vez que temos, por exemplo, um fator inercial de rotação da terra.

* Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?

* \*Medidas do passado podem ter de ser corrigidas por uso corrente quando consideramos, por exemplo, variações no campo magnético da terra? [Pavón-Carrasco, F.J., Marsal, S., Campuzano, S.A., & Torta, J.M. (2021). Signs of a new geomagnetic jerk between 2019 and 2020 from Swarm and observatory data. Earth, Planets and Space, 73, 1-11.]. Isso teria algum valor? 


<br>

<br>

<a name="p3"></a>
<img src="https://github.com/Rogerio-mack/craam/blob/main/figures/Carnegie.png?raw=true" width=300, align="right"> 

## 3. **Análises do Campo Elétrico Atmosférico**

[Back to the top](#top)
<br>

<br>

<br>

<br>


*As medições de gradiente potencial, ou do campo elétrico atmosférico, podem fornecer insights importantes sobre uma série de processos meteorológicos, de relâmpagos, variabilidade climática, poluição por aerossol, neblina e nuvem, bem como influências do clima espacial nos processos atmosféricos. Os trabalhos [1][2][3] empregam a rede
medidas da rede AFINSA (The Atmospheric electric Field Network in South America) de sensores de moinho do campo elétrico (EFM) para obter curvas de variação diária do campo elétrico atmosférico de "tempo bom" e analisar desvios com relação a outros fenômenos como eventos solares (explosões solares e eventos de prótons) e decréscimos Forbush.
Este estudo se propõe a explorar alguns caminhos apontados nessas análises precedentes podendo ser divididos em 3 frentes de trabalho:*

> *1. Analisar outras técnicas de seleção de "tempo bom" e ciclos nas curvas de tempo bom. Particularmente verificar o uso de técnicas de detecção de anomalias fornecendo um método mais automático de tratamento dos dados e técnicas de análise de série temporiais. Aqui poderiam ser utilizados dados da AFINSA, mas também dados da rede GLOCAEM (https://glocaem.wordpress.com/).*

> *2. Analisar o efeito de raios cósmicos [4][5] sobre campo elétrico atmosférico, empregando dados de detectores de raios cósmicos CARPET de CASLEO, como sugere [1], incluindo dados de outras estações disponíveis. Particularmente parece ser de interesse entender o tempo de impacto desses fenômenos sobre o CEA. Sugere-se, aqui, também empregar dados da rede GLOCAEM.*

> *3. Analisar as relações de processos meteorológicos (raios, tempestades etc.) e de poluição por aerossol com variações sobre campo elétrico atmosférico. Seria útil empregar a  medida do CEA como indicador de poluição como sugere [1]? Seria possível e útil empregar esses dados para algum tipo de predição de fenômenos meteorológicos? Sugere-se, aqui, também empregar dados da rede GLOCAEM que, ao que parece, incluem dados meteorológicos.*

### **Referências**

[1] Tacza, J., (2019) Análise da variabilidade do campo elétrico atmosférico durante tempo bom e distúrbios geofísicos. Tese (Doutorado em Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso em: http://tede.mackenzie.br/jspui/handle/tede/3835

[2] Tacza, J., (2015) Análise do campo elétrico atmosférico durante tempo bom e distúrbios geofísicos. Dissertação (Programa de Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso em: http://tede.mackenzie.br/jspui/handle/tede/1302

[3] Tacza, J., Raulin, J.-P., Macotela, E., Marun, A., Fernandez, G., Bertoni, F. C. P., … Makita, K. (2020). Local and global effects on the diurnal variation of the atmospheric electric field in South America by comparison with the Carnegie curve. Atmospheric Research, 104938. doi:10.1016/j.atmosres.2020.10493

[4] Vieira, L. R., (2021). Estudo das séries temporais de raios cósmicos secundários mediante análises por regressão iterativa e transformada wavelet contínua.  Dissertação  (Ciências do Ambiente Solar-Terrestre) – Instituto Nacional de Pesquisas Espaciais, São José dos Campos. Orientadores: Alisson Dal Lago, Nivaor Rodolfo Rigozo e Nelson Jorge Schuch. Acesso: http://urlib.net/sid.inpe.br/mtc-m19/2012/02.08.16.24. 

[5] Mendonça, R.C., Raulin, J., Bertoni, F.C., Echer, E., Makhmutov, V.S., & Fernandez, G.D. (2011). Estudo em múltiplas escalas temporais da intensidade de raios cósmicos medida na superfície terrestre.

### **Fontes de Dados**

[1] AFINSA: (?)

[1] AERONET:  https://aeronet.gsfc.nasa.gov/

[2] CASLEO: https://casleo.conicet.gov.ar/ (open?)

[3] NOAA: https://www.ngdc.noaa.gov/stp/spaceweather.html

[4] GLObal Coordination of Atmospheric Electricity Measurements: https://glocaem.wordpress.com/ (open?)

### **Dúvidas e Pontos Relevantes**

* Pontos a serem melhor entendidos nos trabalhos precedentes (em particular [1], [2] e [3]):

> * Entender o método de análise de épocas superpostas empregado em [1] com a finalidade
de eliminar o ruído de fundo e acentuar possíveis efeitos de baixa amplitude dos eventos solares.

> * Entender a relevância de correção dos dados dos detectores EFM. Não estando instalados no nível do solo precisam ter o valor corrigido. Entretanto, isso foi feito em [1][2] para apenas algumas estações, para outras não. Essa correção seria então tão relevante? 

> * Entender várias periodicidades identificadas ou citadas em [1]. Ciclos de 11 anos foram identificados no caso de manchas solares, e nos dados do campo magnético polar do Sol ciclos de 22 anos. Esses ciclos foram identificados nos dados de entrada? Há algum fenômeno com período de 27 dias? Existem estudos, por exemplo de análise espectral que confirmam esses ciclos observados?

> * O que são o decréscimo Forbush e "Austausch" process? 

> * Segundo Tacza [1], na maior parte dos trabalhos de análise sobre influência de diferentes fenômenos sobre o campo CEA os períodos analisados incluem muitos fenômenos solares, interplanetários e geomagnéticos, como explosões solares, eventos de prótons solares, ejeções de Massa Coronal, Decréscimo Forbush e intensas tempestades geomagnéticas concorrentes e, para entender melhor o papel desses fenômenos no circuito elétrico global, é necessário isolar e estudar separadamente cada um deles. Não é claro, entretanto, como esses feitos foram ou podem ser "isolados", ou se o trabalho se limitou a uma análise individual dos fenômenos sem necessariamente isolar os demais efeitos.

> * Entender métodos empregados em [4] onde séries temporais  de raios cósmicos secundários foram analisadas mediante a aplicação de Análise por Regressão Iterativa (ARIST) e transformada wavelet de Morlet, sendo o ARIST uma técnica espectral clássica que fornece informações globais referentes à frequência, amplitude e fase embutidas em uma série temporal, e por meio da análise wavelet obtém-se a evolução temporal das periodicidades e amplitudes. Isso pode ser útil para outras análises.
  
* Faria algum sentido considerar dados de tempo NÃO BOM? Teria utilidade? O que podemos esperar sobre esses dados ou a inclusão deles aos dados de tempo BOM?

* AFINSA forma parte da rede global para monitoramento do campo elétrico atmosférico chamada GLOCAEM (Global Coordination of Atmospheric Electricity Measurements). O dados Globais também parecem estar disponíveis havendo 17 sites (com a maior parte na europa, seguido do oriente médio, e um na ásia e outro na Antártica). 

* Existem dados de detectores de raios cósmicos CARPET disponíveis além do instalado em Complejo Astronómico El Leoncito (CASLEO)? Southern Space Observatory, in São Martinho da Serra, RS, por exemplo?

* Parece ser relevante estender as análises para dados do GLOCAEM e o período mais recente ([2], por exemplo é de 2015 e o próprio trabalho considerou a janela de tempo empregada pequena).

* Analisar as relações de processos meteorológicos (raios, tempestades etc.) e de poluição por aerossol com variações sobre campo elétrico atmosférico pode ter alguma relevância para o tema de cidades inteligentes e, no momento, um tema de destaque na universidade.

* Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?


<br>

<br>

<a name="p4"></a>
<img src="https://github.com/Rogerio-mack/craam/blob/main/figures/EarthquakeVLF.png?raw=true" width=300, align="right"> 

## 4. **Análise das Variações de Fase nas Ondas de VLF nos Períodos que Precedem Terromotos**

[Back to the top](#top)

<br>

<br>

<br>

<br>


*Uma série de fenômenos físicos transientes que alteram a condutividade da baixa ionosfera o que provoca variações da fase e amplitude de sinais de muito baixa frequência (VLF) [1]. Dentre esses fenômenos encontram-se o ciclo diurno, fases do ciclo solar, variações no campo magnético, condições atmosféricas [2][3][4] mas também fenômenos sísmicos [1][5][6]. Este estudo, a exemplo dos trabalhos anteriores de Macoleta (2015) [1], Grant (2013) [5] e  Hayakawa (2011) [6], busca investigar as relações das variações de fase de sinais VLF com base em eventos sísmicos mais recentes sendo a técnica relevante para predição desses eventos.* 

### **Referências**

[1] Macoleta, E.L., (2015) Contribuição ao estudo de distúrbios ionosféricos utilizando a técnica de VLF. Dissertação (Programa de Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso em: http://tede.mackenzie.br/jspui/handle/tede/1301

[2] Raulin, J.P. (2011). The South America VLF Network (SAVNET): Providing new ground-based diagnostics of space weather conditions. 2011 XXXth URSI General Assembly and Scientific Symposium, 1-4.

[3] Crombie, D.D. (1964). Periodic fading of VLF signals received over long paths during sunrise and sunset. Journal of Research of the National Bureau of Standards, Section D: Radio Science, 27.

[4] Pacini, A.A., & Raulin, J.P. (2006). Solar X-ray flares and ionospheric sudden phase anomalies relationship : A solar cycle phase dependence. Journal of Geophysical Research, 111.

[5] Grant, R.A., Raulin, J.P., & Freund, F. (2013). Camera trap records of animal activity prior to a M=7 earthquake in Northern Peru.

[6] Hayakawa, M., Raulin, J.P., Kasahara, Y., Bertoni, F.C., Hobara, Y., & Guevara-Day, W. (2011). Ionospheric perturbations in possible association with the 2010 Haiti earthquake, as based on medium-distance subionospheric VLF propagation data. Natural Hazards and Earth System Sciences, 11, 513-518.

### **Fontes de Dados**

[1] SAVNET:  (?)

[2] Sunspot index and Long-term Solar Observations: https://wwwbis.sidc.be/silso/

[3] Base de Dados:  https://earthquake.usgs.gov/earthquakes/search/

### **Dúvidas e Pontos Relevantes**

* Parece relevante empregar dados de terremotos mais recentes uma vez que os estudos anteriores tratam somente eventos de 2015 para trás e outros eventos como erupções parecem ter potencial de serem explorados.

* Pontos a entender no estudo de Macoleta (2015) [1]:

> * Por que se foram selecionados somente eventos com Richter 7 ou superior e por que, aparentemente, foram analisados somente 2 eventos.

> * Um ponto importante: O estudo alerta para necessidade de verificar variações dos sinais fora da janela de eventos sísmicos. 

> * Entender melhor o problema a necessidade de "aprimorar um método de normalização independente da órbita de de propagação", e este pode ser um resultado a ser buscado no estudo até em substituição ao objetivo principal.

> * Como em outros trabalhos [1] também emprega a transformada wavelet de Morlet que precisa ser melhor compreendida aqui.

* Importante buscar as relações entre o período e a intensidade da variação de fase, e a intensidade do evento sísmico, isto é, qual o tempo que a variação antecede o terromoto a depender de sua intensidade? 

* Potencialmente importante é verificar a diferença de efeito de fase em estações receptoras à diferentes distâncias do epicentro.

* Verificar na base de dados https://earthquake.usgs.gov/earthquakes/search/ scripts disponíveis para extração dos dados.

* Existem outras fontes/redes de dados VLF disponíveis?

* Verificar. Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?


<br>

<br>

<a name="p5"></a>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/SST_with_open_radome.jpg/450px-SST_with_open_radome.jpg" width=250, align="right"> 

## 5. **Exploração de dados de Ondas Sub-milimétricas em  Fenômenos Astrofísicos**

[Back to the top](#top)

<br>

<br>

<br>

<br>


*As observações dos comprimentos de onda sub-milimétricos são importantes porque nos permitem estudar diversos objetos astrofísicos com o sol, planetas, região do centro da galáxia, nuvens moleculares etc. e permitem compreender melhor os processos físicos envolvidos [1]. Vários estudos têm sido feitos empregando ondas sub-milimétricos para análise de explosões solares buscando explicar os fenômenos que envolvem as partículas solares [2][3] havendo várias questões que ainda permanecem em aberto [4]. Este estudo, a exploração dos dados de Ondas Sub-milimétricas, apresenta potencialmente 2 frentes de trabalho:*

> *1. Em Espinoza (2022) [1] são analisdados diversos aspectos a influência da opacidade atmosférica sobre os sinais de ondas sub-milimétricos. Nesta frente a ideia é aplicar um método, baseado em [1], para 'corrigir' a influência da opacidade nos sinais e recuperar a verdadeira temperatura de brilho da fonte em eventos de explosões solares e outros (a definir quais).*

> *2. Seguindo [3][4], parece ser de interesse analisar a ocorrência de diferentes comprimentos de onda em fenômenos astrofísicos. Aqui pode-se explorar eventos de explosões solares empregando faixas específicas de onda ou ainda outros fenômenos.*


### **Referências**

[1] Espinoza, D.V.C. (2022) Opacidade Atmosférica em Ondas Sub-milimétricas: O papel do Conteúdo de vapor de água (PWV). Tese (Programa de Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso (provisório) em: https://github.com/Rogerio-mack/craam/blob/main/articles/Tese_DeysiCornejo%20(1).pdf

[2] Giménez de Castro, C.G., Raulin, J.P., Valle Silva, J.F., Simões, P.J., Kudaka, A.S., & Valio, A. (2018). The 6 September 2017 X9 Super Flare Observed From Submillimeter to Mid‐IR. Space Weather.

[3] Trottet, G., Raulin, J.P., MacKinnon, A.L., Giménez de Castro, G., Simões, P.J., Cabezas, D.P., Luz, V.D., Luoni, M.L., & Kaufmann, P. (2015). Origin of the 30 THz Emission Detected During the Solar Flare on 2012 March 13 at 17:20 UT. Solar Physics, 290, 2809-2826.

[4] Krucker, S., Giménez de Castro, C.G., Hudson, H.S., Trottet, G., Bastian, T.S., Hales, A.S., Kašparová, J., Klein, K.-., Kretzschmar, M., Lüthi, T., MacKinnon, A.L., Pohjolainen, S., & White, S.D. (2013). Solar flares at submillimeter wavelengths. The Astronomy and Astrophysics Review, 21, 1-45.

### **Fontes de Dados**

[1] AERONET:  https://aeronet.gsfc.nasa.gov/

[2] CASLEO: https://casleo.conicet.gov.ar/ (open?)

[3] NOAA: https://www.ngdc.noaa.gov/stp/spaceweather.html


### **Dúvidas e Pontos Relevantes**

* Pontos a entender no estudo de Espinoza (2022) [1]:

> * O trabalho mostra em diversos aspectos a influência da opacidade atmosférica. Mas realmente propõe um método prático para 'corrigir' essa influência da opacidade nos sinais e recuperar a verdadeira temperatura de brilho da fonte? A primeira frente proposta se baseia de que há aqui um método prático a ser aplicado.

> * A confirmar, mas não parece fazer sentido, ou mesmo útil, explorar outras possíveis fontes de atenuação ou perturbação das ondas sub-milimétricas, como a poluição.

* Dúvidas gerais sobre ondas sub-milimétricas:

> * Segundo [List of solar telescopes](https://en.wikipedia.org/wiki/List_of_solar_telescopes#cite_note-31) o telescópio SST de CASLEO é o único atualmente em operação. O CSO (Caltech Submillimeter Observatory) aparentemente foi decomissionado. Isso reflete uma retração de análises a partir de ondas sub-milimétricas por outros tipos de análise? Que aspectos só poderiam ser explorados por esse tipo de onda.

> * Existem vários outros rádio telescópios. O Atacama Large Millimeter/submillimeter Array (ALMA), aparentemente pode capturar 'imagens' (ondas?) entre  0.4 mm to 3 mm. Não é claro, entretanto, a diferença do radio telescópio de CASLEO para esses outros rádios telescópios. Por exemplo o ALMA opera com várias antenas e não é claro se ondas entre 0.4 mm to 3 mm são ondas submilimétricas ou não (em [4], *solar
flares at submillimeter wavelengths, defined here as observing wavelengths shorter
than 3 mm (frequencies higher than 0.1 THz)*).

* De [4], *We generally
have lacked systematic observations in the millimeter-wave to far-infrared range that
are needed to complete our picture of these events, and encourage observations with
new facilities.* Não é claro a diferença entre ondas submilimétricas e miliméticas e quais de fato carecem observações. Aparentemente, também, o surgimento dessas ondas é um fenômeno ainda pouco entendido.

* O estudo de Trottet, G. (2015) [3] parece explorar um evento específico (SOL2012-03-13) em vários comprimentos de onda. Todos parecem concordar com variações correlacionadas ao evento da explosão solar. Aparentemente, então, quaisquer comprimentos de onda podem ser empregados para 'detectar' o evento. Qual deveria ser então empregado? A análise dos  diferentes comprimentos de onda permitiriam tirar que potencias conclusões? Em princípio, essas análises permitem tirar conclusões sobre que fenômenos físicos no sol dão origem a esses diferentes comprimentos de onda exigindo um conhecimento bastante profundo da física solar e suas partículas.

* Temos acesso a dados de outros rádio telescópios como o ALMA por exemplo? 

* Verificar. Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* Os dados do rádio telescópio poderiam ser empregados para produzir 'imagens' úteis do sol?

