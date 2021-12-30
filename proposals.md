# Potenciais Propostas de Trabalho

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

* Alguma métrica das explosões solares (por exemplo as classes A, B, C, M, X) ou apenas identificar sua ocorrência?

* Empregar dados de fase, amplitude ou ambos?

* Potenciais modelos de rede: LSTM, convolucionais, modelos para detecção de anomalia.

* Relevância. Identificação em tempo real seria *i.* viável e *ii.* relevante neste caso?

* Verificar. Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

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

* Verificar. Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?

* \*Medidas do passado podem ter de ser corrigidas por uso corrente quando consideramos, por exemplo, variações no campo magnético da terra? [Pavón-Carrasco, F.J., Marsal, S., Campuzano, S.A., & Torta, J.M. (2021). Signs of a new geomagnetic jerk between 2019 and 2020 from Swarm and observatory data. Earth, Planets and Space, 73, 1-11.]. Isso teria algum valor? 

**Potenciais Referências**

[1] Lambeck, K. (1980). Changes in length-of-day and atmospheric circulation. Nature, 286(5769), 104–105. doi:10.1038/286104a0.

[2] Menezes G.O.,Raulin, J.P., Ramirez, R.F.H., Silva, L.A., Pamboukian, S.V.D., Merkowitz, S. (2022) Forecasting of Space Geodesy Data and Investigation of the Relationship with the Solar Activity. [*to appear*](https://github.com/Rogerio-mack/craam/blob/main/articles/article_Guilherme_projeto_Novo.pdf)

[3] Malkin, Z. (2009). On comparison of the Earth orientation parameters obtained from different VLBI networks and observing programs. Journal of Geodesy, 83, 547-556.

[4] Modiri, S., Belda, S., Hoseini, M., Heinkelmann, R., Ferrándiz, J.M., & Schuh, H. (2020). Um novo método híbrido para melhorar a previsão de ultracurto prazo de LOD. Journal of Geodesy, 94

[5] Liao, D., Wang, Q.J., Zhou, Y., Liao, X., & Huang, C. (2012). Long-term prediction of the Earth Orientation Parameters by the artificial neural network technique. Journal of Geodynamics, 62, 87-92.

[6] Goodfellow, I., Bengio, Y., & Courville, A (2016). Deep Learning. MIT Press. Also available online: http://www.deeplearningbook.org.


<br>

<br>

<img src="https://github.com/Rogerio-mack/craam/blob/main/figures/Carnegie.png?raw=true" width=300, align="right"> 

## 3. **Análises do Campo Elétrico Atmosférico**

<br>

<br>

<br>

<br>


*As medições de gradiente potencial, ou do campo elétrico atmosférico, podem fornecer insights importantes sobre uma série de processos meteorológicos, de relâmpagos, variabilidade climática, poluição por aerossol, neblina e nuvem, bem como influências do clima espacial nos processos atmosféricos. Os trabalhos [1][2][3] empregam a rede
medidas da rede AFINSA (The Atmospheric electric Field Network in South America) de sensores de moinho do campo elétrico (EFM) para obter curvas de variação diária do campo elétrico atmosférico de "tempo bom" e analisar desvios com relação a outros fenômenos como eventos solares (explosões solares e eventos de prótons) e decréscimos Forbush.
Este estudo se propõe a explorar alguns caminhos apontados nessas análises precedentes podendo ser divididos em 3 frentes de trabalho: 

> 1. Analisar outras técnicas de seleção de "tempo bom" e ciclos nas curvas de tempo bom. Particularmente verifica o uso de técnicas de detecção de anomalias fornecendo um método mais automático de tratamento dos dados e técnicas de análise de série temporiais. Aqui poderiam ser utilizados dados da AFINSA, mas também dados da rede GLOCAEM (https://glocaem.wordpress.com/).

> 2. Analisar o efeito de raios cósmicos [4][5] sobre campo elétrico atmosférico, empregando dados de detectores de raios cósmicos CARPET de CASLEO, como sugere [1], incluindo dados de outras estações disponíveis. Particularmente parece ser de interesse entender o tempo de impacto desses fenômenos sobre o CEA. Sugere-se, aqui, também empregar dados da rede GLOCAEM.

> 3. Analisar as relações de processos meteorológicos (raios, tempestades etc.) e de poluição por aerossol com variações sobre campo elétrico atmosférico. Seria útil empregar a  medida do CEA como indicador de poluição como sugere [1]? Seria possível e útil empregar esses dados para algum tipo de predição de fenômenos meteorológicos? Sugere-se, aqui, também empregar dados da rede GLOCAEM que, ao que parece, incluem dados meteorológicos.

**Dúvidas e Pontos Relevantes**

* Pontos a serem melhor entendidos nos trabalhos precedentes (em particular [1], [2] e [3]):

> * Entender o método de análise de épocas superpostas empregado em [1] com a finalidade
de eliminar o ruído de fundo e acentuar possíveis efeitos de baixa amplitude dos eventos solares.

> * Entender a relevância de correção dos dados dos detectores EFM. Não estando instalados no nível do solo precisam ter o valor corrigido. Entretanto, isso foi feito em [1][2] para apenas algumas estações, para outras não. Essa correção seria então tão relevante? 

> * Entender várias periodicidades identificadas ou citadas em [1]. Ciclos de 11 anos foram identificados no caso de manchas solares, e nos dados do campo magnético polar do Sol ciclos de 22 anos. Esses ciclos foram identificados nos dados de entrada? Há açgum fenômeno com período de 27 dias? Existem estudos, por exemplo de análise espectral que confirmam esses ciclos observados?

> * O que são o decréscimo Forbush e "Austausch" process? 

> * Segundo Tacza [1], na maior parte dos trabalhos de análise sobre influência de diferentes fenômenos sobre o campo CEA os períodos analisados incluem muitos fenômenos solares, interplanetários e geomagnéticos, como explosões solares, eventos de prótons solares, ejeções de Massa Coronal, Decréscimo Forbush e intensas tempestades geomagnéticas concorrentes e, para entender melhor o papel desses fenômenos no circuito elétrico global, é necessário isolar e estudar separadamente cada um deles. Não é claro, entretanto, como esses feitos foram ou podem ser "isolados", ou se o trabalho se limitou a uma análise individual dos fenômenos sem necessariamente isolar os demais efeitos.

> * Entender métodos empregados em [4] onde séries temporais  de raios cósmicos secundários foram analisadas mediante a aplicação de Análise por Regressão Iterativa (ARIST) e transformada wavelet de Morlet, sendo o ARIST uma técnica espectral clássica que fornece informações globais referentes à frequência, amplitude e fase embutidas em uma série temporal, e por meio da análise wavelet obtém-se a evolução temporal das periodicidades e amplitudes. Isso pode ser útil para outras análises.
  
* Faria algum sentido considerar dados de tempo NÃO BOM? Teria utilidade? O que podemos esperar sobre esses dados ou a inclusão deles aos dados de tempo BOM?

* AFINSA forma parte da rede global para monitoramento do campo elétrico atmosférico chamada GLOCAEM (Global Coordination of Atmospheric Electricity Measurements). O dados Globais também parecem estar disponíveis havendo 17 sites (com a maior parte na europa, seguido do oriente médio, e um na ásia e outro na Antártica). 

* Existem dados de detectores de raios cósmicos CARPET disponíveis além do instalado em Complejo Astronómico El Leoncito (CASLEO)? Southern Space Observatory, in São Martinho da Serra, RS, por exemplo?

* Parece ser relevante estender as análises para dados do GLOCAEM e o período mais recente ([2], por exemplo é de 2015 e o próprio trabalho considerou a janela de tempo empregada pequena).

* Analisar as relações de processos meteorológicos (raios, tempestades etc.) e de poluição por aerossol com variações sobre campo elétrico atmosférico pode ter alguma relevância para o tema de cidades inteligentes e, no momento, um tema de destaque na universidade.

* Verificar. Faz sentido disponibilizar um repositório com acesso mais facilitado aos pesquisadores do CRAAM ou com acesso público?

* \*Existe um modelo matemático, um conjunto de equações ou mesmo variáveis que possam ser empregados para análise de fatores?

**Potenciais Referências**

[1] Tacza, J., (2019) Análise da variabilidade do campo elétrico atmosférico durante tempo bom e distúrbios geofísicos. Tese (Doutorado em Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso em: http://tede.mackenzie.br/jspui/handle/tede/3835

[2] Tacza, J., (2015) Análise do campo elétrico atmosférico durante tempo bom e distúrbios geofísicos. Dissertação (Programa de Ciências e Aplicaçőes Geoespaciais) - Universidade Presbiteriana Mackenzie, São Paulo. Orientador: Jean-Pierre Raulin. Acesso em: http://tede.mackenzie.br/jspui/handle/tede/1302

[3] Tacza, J., Raulin, J.-P., Macotela, E., Marun, A., Fernandez, G., Bertoni, F. C. P., … Makita, K. (2020). Local and global effects on the diurnal variation of the atmospheric electric field in South America by comparison with the Carnegie curve. Atmospheric Research, 104938. doi:10.1016/j.atmosres.2020.10493

[4] Vieira, L. R., (2021). Estudo das séries temporais de raios cósmicos secundários mediante análises por regressão iterativa e transformada wavelet contínua.  Dissertação  (Ciências do Ambiente Solar-Terrestre) – Instituto Nacional de Pesquisas Espaciais, São José dos Campos. Orientadores: Alisson Dal Lago, Nivaor Rodolfo Rigozo e Nelson Jorge Schuch. Acesso: http://urlib.net/sid.inpe.br/mtc-m19/2012/02.08.16.24. 

[5] Mendonça, R.C., Raulin, J., Bertoni, F.C., Echer, E., Makhmutov, V.S., & Fernandez, G.D. (2011). Estudo em múltiplas escalas temporais da intensidade de raios cósmicos medida na superfície terrestre.

