using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;

namespace Megasena.Enhanced
{
    public class EnhancedPredictor
    {
        // Número total de dezenas na Mega-Sena
        private const int NUMBER_RANGE = 60;

        // Tamanho principal da janela que vamos usar
        private readonly int _historyWindow = 100;

        // Sub-janelas para extrair estatísticas em múltiplos horizontes
        private readonly int[] _subWindows = new[] { 10, 20, 50, 100 };

        /// <summary>
        /// Classe interna para encapsular a rede neural e o salvamento do melhor estado
        /// </summary>
        private class EnhancedNetwork
        {
            private readonly BasicNetwork _network;
            private readonly int _inputSize;

            // Mantém a melhor configuração já encontrada
            private double _bestError;
            private BasicNetwork _bestNetwork;

            public EnhancedNetwork(int inputSize)
            {
                _inputSize = inputSize;
                _network = CreateNetwork();
                _bestNetwork = CreateNetwork();
                _bestError = double.MaxValue;
            }

            /// <summary>
            /// Cria a estrutura da rede com:
            /// - Camada de entrada: inputSize neurônios
            /// - Camadas ocultas com ReLU
            /// - Camada de saída: 60 neurônios, Sigmoid (probabilidade independente)
            /// </summary>
            private BasicNetwork CreateNetwork()
            {
                var OUTPUT_SIZE = NUMBER_RANGE; // 60 saídas

                var network = new BasicNetwork();

                // Camada de entrada
                network.AddLayer(new BasicLayer(null, true, _inputSize));

                // Camadas intermediárias com ReLU
                network.AddLayer(new BasicLayer(new ActivationReLU(), true, _inputSize * 2));
                network.AddLayer(new BasicLayer(new ActivationReLU(), true, _inputSize * 2));
                network.AddLayer(new BasicLayer(new ActivationReLU(), true, _inputSize));

                // Camada de saída: 60 neurônios, Sigmoid (probabilidade 0..1)
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, OUTPUT_SIZE));

                network.Structure.FinalizeStructure();
                network.Reset();

                return network;
            }

            /// <summary>
            /// Treina a rede usando ResilientPropagation, salvando pesos sempre que achar uma melhora
            /// no erro. Focamos em quantidade grande de épocas (p. ex. 30.000) para buscar maior convergência.
            /// </summary>
            public void Train(IMLDataSet trainingSet, int maxEpochs)
            {
                var train = new ResilientPropagation(_network, trainingSet);
                train.NumThreads = Environment.ProcessorCount;

                int epochsWithoutImprovement = 0;

                for (int epoch = 0; epoch < maxEpochs; epoch++)
                {
                    train.Iteration();

                    // Se melhorou no erro, salvamos estado
                    if (train.Error < _bestError)
                    {
                        _bestError = train.Error;

                        // Copia pesos para _bestNetwork
                        for (int i = 0; i < _network.LayerCount - 1; i++)
                        {
                            for (int fromNeuron = 0; fromNeuron < _network.GetLayerNeuronCount(i); fromNeuron++)
                            {
                                for (int toNeuron = 0; toNeuron < _network.GetLayerNeuronCount(i + 1); toNeuron++)
                                {
                                    double weight = _network.GetWeight(i, fromNeuron, toNeuron);
                                    _bestNetwork.SetWeight(i, fromNeuron, toNeuron, weight);
                                }
                            }
                        }

                        epochsWithoutImprovement = 0;
                    }
                    else
                    {
                        epochsWithoutImprovement++;
                        // Se ficarmos muitas épocas sem melhora, encerramos
                        if (epochsWithoutImprovement > 500)
                            break;
                    }

                    // Se o erro estiver muito baixo, podemos encerrar
                    if (train.Error < 0.00001)
                        break;

                    // Log assíncrono para acompanhar o progresso
                    if (epoch % 100 == 0) // Log a cada 100 épocas
                    {
                        Console.WriteLine($"[Training] Época {epoch}: Erro = {train.Error}");
                    }
                }

                // Restaura o melhor estado
                for (int i = 0; i < _network.LayerCount - 1; i++)
                {
                    for (int fromNeuron = 0; fromNeuron < _network.GetLayerNeuronCount(i); fromNeuron++)
                    {
                        for (int toNeuron = 0; toNeuron < _network.GetLayerNeuronCount(i + 1); toNeuron++)
                        {
                            double weight = _bestNetwork.GetWeight(i, fromNeuron, toNeuron);
                            _network.SetWeight(i, fromNeuron, toNeuron, weight);
                        }
                    }
                }
            }

            /// <summary>
            /// Retorna o array de 60 probabilidades (uma para cada número),
            /// dada uma entrada de features.
            /// </summary>
            public double[] Predict(IMLData input)
            {
                return ((BasicMLData)_network.Compute(input)).Data;
            }
        }

        /// <summary>
        /// Ponto de entrada para prever os próximos 6 números,
        /// usando uma rede neural mais complexa e com várias janelas internas.
        /// </summary>
        public List<int> PredictNextNumbers(List<MegasenaResult> history)
        {
            // Garante que temos pelo menos _historyWindow sorteios
            if (history.Count < _historyWindow)
            {
                Console.WriteLine($"[EnhancedPredictor] AVISO: poucos sorteios no histórico ({history.Count}). " +
                                  $"Ideal >= {_historyWindow} para melhor resultado.");
            }

            // Extrai features para a janela "mais recente"
            var latestWindow = history.Take(_historyWindow).ToList();
            var features = ExtractFeaturesMultiWindow(latestWindow);
            var inputData = new BasicMLData(features);

            // Cria a rede e prepara dataset de treino completo
            var network = new EnhancedNetwork(features.Length);
            var trainingSet = PrepareTrainingSet(history);

            Console.WriteLine("\n[EnhancedPredictor] Iniciando treinamento avançado com múltiplas janelas...\n");
            // Aumente bastante a quantidade de épocas
            network.Train(trainingSet, 30000);

            // Obtemos as probabilidades de cada número (0..1)
            var prediction = network.Predict(inputData);

            // Seleciona os 6 números de maior probabilidade
            var top6 = ConvertPredictionToNumbers(prediction);

            Console.WriteLine("[EnhancedPredictor] Previsão concluída.\n");

            return top6;
        }

        /// <summary>
        /// Extrai features considerando múltiplas janelas (10, 20, 50, 100) 
        /// para cada jogo passado, gerando uma grande lista de variáveis:
        ///  - Freq de cada número em cada sub-janela
        ///  - Última ocorrência de cada número em cada sub-janela
        /// </summary>
        private double[] ExtractFeaturesMultiWindow(List<MegasenaResult> mainWindow)
        {
            // Colecionamos as features de cada sub-janela em uma lista única
            var combinedFeatures = new List<double>();

            // Processar cada sub-janela em paralelo
            var subFeatures = _subWindows.AsParallel().Select(w =>
            {
                var wSize = Math.Min(w, mainWindow.Count);
                var partialWindow = mainWindow.Take(wSize).ToList();

                var freq = CalculateFrequency(partialWindow);
                var lastOcc = CalculateLastOccurrence(partialWindow);

                return new List<double>(freq).Concat(lastOcc).ToList();
            }).ToList();

            // Combinar todas as features das sub-janelas
            foreach (var sub in subFeatures)
            {
                combinedFeatures.AddRange(sub);
            }

            return combinedFeatures.ToArray();
        }

        /// <summary>
        /// Calcula frequência normalizada (0..1) de cada número (1..60)
        /// na lista de sorteios passada.
        /// </summary>
        private double[] CalculateFrequency(List<MegasenaResult> window)
        {
            double[] freq = new double[NUMBER_RANGE];
            foreach (var draw in window)
            {
                freq[draw.V1 - 1]++;
                freq[draw.V2 - 1]++;
                freq[draw.V3 - 1]++;
                freq[draw.V4 - 1]++;
                freq[draw.V5 - 1]++;
                freq[draw.V6 - 1]++;
            }

            // Normaliza pela quantidade de sorteios usando paralelização
            Parallel.For(0, NUMBER_RANGE, i =>
            {
                freq[i] /= (double)window.Count;
            });

            return freq;
        }

        /// <summary>
        /// Calcula a última ocorrência (0..1) de cada número (1..60) na lista.
        ///  - 0 significa que apareceu no sorteio mais recente.
        ///  - 1 significa que não apareceu em nenhum sorteio dessa janela.
        ///  - Valores intermediários indicam a posição relativa da última ocorrência.
        /// </summary>
        private double[] CalculateLastOccurrence(List<MegasenaResult> window)
        {
            // Preenche com o valor "window.Count" (significa que não apareceu)
            double[] lastOcc = Enumerable.Repeat((double)window.Count, NUMBER_RANGE).ToArray();

            // Processar cada sorteio e atualizar a última ocorrência
            for (int i = 0; i < window.Count; i++)
            {
                var draw = window[i];
                var numbers = new[] { draw.V1, draw.V2, draw.V3, draw.V4, draw.V5, draw.V6 };

                foreach (var n in numbers)
                {
                    // Atualiza apenas se ainda não foi atualizado
                    if (lastOcc[n - 1] == window.Count)
                    {
                        // (window.Count - i - 1) representa quantos sorteios atrás apareceu
                        lastOcc[n - 1] = (double)(window.Count - i - 1);
                    }
                }
            }

            // Normaliza pelo total de sorteios nessa janela usando paralelização
            Parallel.For(0, NUMBER_RANGE, i =>
            {
                lastOcc[i] /= (double)window.Count;
            });

            return lastOcc;
        }

        /// <summary>
        /// Prepara o dataset de treinamento usando todo o histórico disponível.
        /// Cada exemplo será: (features extraídas das janelas passadas) -> (vetor 60 com 1/0).
        /// </summary>
        private IMLDataSet PrepareTrainingSet(List<MegasenaResult> fullHistory)
        {
            var inputList = new ConcurrentBag<double[]>();
            var outputList = new ConcurrentBag<double[]>();

            // Utilizar Parallel.For para acelerar a preparação dos dados
            Parallel.For(_historyWindow, fullHistory.Count, i =>
            {
                // Seleciona a "janela principal" de i-_historyWindow até i-1
                var windowData = fullHistory.Skip(i - _historyWindow).Take(_historyWindow).ToList();

                // Extrai features (múltiplas sub-janelas)
                var features = ExtractFeaturesMultiWindow(windowData);

                // Cria a saída de 60 elementos (1 se o número apareceu no sorteio i, senão 0)
                var outputs = new double[NUMBER_RANGE];
                Array.Fill(outputs, 0.0);

                var currentDraw = fullHistory[i];
                outputs[currentDraw.V1 - 1] = 1;
                outputs[currentDraw.V2 - 1] = 1;
                outputs[currentDraw.V3 - 1] = 1;
                outputs[currentDraw.V4 - 1] = 1;
                outputs[currentDraw.V5 - 1] = 1;
                outputs[currentDraw.V6 - 1] = 1;

                // Adicionar aos bags de forma thread-safe
                inputList.Add(features);
                outputList.Add(outputs);
            });

            // Converter ConcurrentBag para arrays
            var inputArray = inputList.ToArray();
            var outputArray = outputList.ToArray();

            return new BasicMLDataSet(inputArray, outputArray);
        }

        /// <summary>
        /// Converte o array de 60 probabilidades em 6 números,
        /// pegando os índices das 6 maiores probabilidades.
        /// </summary>
        private List<int> ConvertPredictionToNumbers(double[] prediction)
        {
            // Cria uma lista (numero, prob)
            var probList = new List<(int num, double prob)>();
            for (int i = 0; i < NUMBER_RANGE; i++)
            {
                probList.Add((i + 1, prediction[i]));
            }

            // Ordena do maior para o menor
            var ordered = probList.OrderByDescending(x => x.prob).ToList();

            // Pega os 6 mais prováveis
            var top6 = ordered.Take(6).Select(x => x.num).OrderBy(x => x).ToList();

            return top6;
        }
    }
}
