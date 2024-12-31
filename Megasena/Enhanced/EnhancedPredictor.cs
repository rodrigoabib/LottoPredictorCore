using System;
using System.Collections.Generic;
using System.Linq;
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
        // Número total de bolas possíveis (1 a 60).
        private const int NUMBER_RANGE = 60;

        // Tamanho da janela de histórico que vamos analisar (pode aumentar se quiser).
        private readonly int _historyWindow = 50;

        // Classe interna que encapsula a rede neural “melhorada”
        private class EnhancedNetwork
        {
            private readonly BasicNetwork _network;
            private readonly int _inputSize;

            // Salva estado da melhor rede
            private double _bestError;
            private BasicNetwork _bestNetwork;

            public EnhancedNetwork(int inputSize)
            {
                _inputSize = inputSize;
                _network = CreateNetwork();
                // A _bestNetwork inicia igual à rede base
                _bestNetwork = CreateNetwork(); 
                _bestError = double.MaxValue;
            }

            /// <summary>
            /// Cria a estrutura da rede com 60 neurônios de saída,
            /// um para cada número da loteria (1..60).
            /// </summary>
            private BasicNetwork CreateNetwork()
            {
                var OUTPUT_SIZE = NUMBER_RANGE;  // 60 saídas

                var network = new BasicNetwork();

                // Camada de entrada
                network.AddLayer(new BasicLayer(null, true, _inputSize));

                // Camadas intermediárias (você pode ajustar quantas quiser)
                network.AddLayer(new BasicLayer(new ActivationTANH(), true, _inputSize * 2));
                network.AddLayer(new BasicLayer(new ActivationTANH(), true, _inputSize * 2));
                network.AddLayer(new BasicLayer(new ActivationTANH(), true, _inputSize));

                // Camada de saída: 60 neurônios, cada um representando a probabilidade
                // de saída de um número (1..60), todos com Sigmoid (intervalo 0..1).
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, OUTPUT_SIZE));

                network.Structure.FinalizeStructure();
                network.Reset();

                return network;
            }

            /// <summary>
            /// Treina a rede com ResilientPropagation, salvando sempre que achar um estado melhor.
            /// A quantidade de épocas pode ser bem alta para maximizar a chance de convergência.
            /// </summary>
            public void Train(IMLDataSet trainingSet, int maxEpochs)
            {
                var train = new ResilientPropagation(_network, trainingSet);
                train.NumThreads = Environment.ProcessorCount;

                // Se desejado, podemos soltar logs mais detalhados de tempos em tempos
                int epochsWithoutImprovement = 0;
                for (int epoch = 0; epoch < maxEpochs; epoch++)
                {
                    train.Iteration();

                    // Verifica se melhorou de erro
                    if (train.Error < _bestError)
                    {
                        _bestError = train.Error;
                        
                        // Salva pesos da _network em _bestNetwork
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
                        // Pode aumentar esse limite para deixar a rede insistir mais
                        if (epochsWithoutImprovement > 500) 
                            break;
                    }

                    // Se o erro chegou a um patamar muito baixo, podemos parar
                    if (train.Error < 0.00001)
                        break;
                }

                // Restaura a melhor configuração
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
            /// Gera a previsão (vetor de 60 probabilidades, uma para cada número)
            /// a partir de um vetor de entrada (features).
            /// </summary>
            public double[] Predict(IMLData input)
            {
                return ((BasicMLData)_network.Compute(input)).Data;
            }
        }

        /// <summary>
        /// Método chamado para prever os próximos 6 números.
        /// </summary>
        /// <param name="history">
        /// Lista de resultados antigos (do mais recente para o mais antigo ou vice-versa).
        /// </param>
        public List<int> PredictNextNumbers(List<MegasenaResult> history)
        {
            // Extrai as features da "janela" mais recente de 50 (ou quantos tivermos)
            var features = ExtractFeatures(history.Take(_historyWindow).ToList());
            var inputData = new BasicMLData(features);

            // Cria a rede e o dataset
            var network = new EnhancedNetwork(features.Length);
            var trainingSet = PrepareTrainingSet(history, features.Length);

            // Treina por bastante tempo (por ex., 20.000 épocas) para garantir maior convergência
            Console.WriteLine("\n[EnhancedPredictor] Iniciando treinamento aprofundado...\n");
            network.Train(trainingSet, 20000);

            // Agora obtemos a previsão: 60 probabilidades (cada índice corresponde a "número index+1")
            var prediction = network.Predict(inputData);

            // Seleciona os 6 números de maior probabilidade
            return ConvertPredictionToNumbers(prediction);
        }

        /// <summary>
        /// Extrai as features que vão alimentar a rede. Você pode adicionar mais métricas se quiser.
        /// </summary>
        private double[] ExtractFeatures(List<MegasenaResult> history)
        {
            // Frequências
            var freq = new double[NUMBER_RANGE];
            foreach (var result in history)
            {
                freq[result.V1 - 1]++;
                freq[result.V2 - 1]++;
                freq[result.V3 - 1]++;
                freq[result.V4 - 1]++;
                freq[result.V5 - 1]++;
                freq[result.V6 - 1]++;
            }

            // Normaliza a frequência pelo tamanho do history
            for (int i = 0; i < NUMBER_RANGE; i++)
                freq[i] /= history.Count;

            // Últimas ocorrências
            var lastOccurrence = new double[NUMBER_RANGE];
            // Preenche com um valor grande (por ex. history.Count), caso o número não tenha aparecido
            Array.Fill(lastOccurrence, (double)history.Count);

            // Calcula a posição da última ocorrência
            for (int i = history.Count - 1; i >= 0; i--)
            {
                var numbers = new[] { history[i].V1, history[i].V2, history[i].V3,
                                      history[i].V4, history[i].V5, history[i].V6 };

                foreach (var num in numbers)
                {
                    // Se ainda estiver com valor "history.Count", atualiza
                    if (lastOccurrence[num - 1] == history.Count)
                    {
                        lastOccurrence[num - 1] = history.Count - i - 1;
                    }
                }
            }

            // Normaliza a última ocorrência pelo tamanho do history
            for (int i = 0; i < NUMBER_RANGE; i++)
                lastOccurrence[i] /= history.Count;

            // Combine as duas features num vetor único
            var combined = new List<double>();
            combined.AddRange(freq);
            combined.AddRange(lastOccurrence);

            // Caso queira adicionar mais estatísticas (por ex. média de “intervalo entre aparições”,
            // sumário de hot streaks etc.), faça aqui.

            return combined.ToArray();
        }

        /// <summary>
        /// Prepara o dataset de treinamento usando toda a base de histórico.
        /// Vamos gerar, para cada ponto no histórico (a partir de _historyWindow), 
        /// um vetor de entrada (as features da janela anterior) e um vetor de saída (60 posições, 0 ou 1).
        /// </summary>
        private IMLDataSet PrepareTrainingSet(List<MegasenaResult> history, int featureSize)
        {
            var inputList = new List<double[]>();
            var outputList = new List<double[]>();

            // Do _historyWindow até o final, geramos exemplos
            // Exemplo: se temos 200 sorteios, e _historyWindow=50, geramos 150 exemplos
            for (int i = _historyWindow; i < history.Count; i++)
            {
                // Seleciona a janela de i-50 até i-1, inclusive
                var windowHistory = history.Skip(i - _historyWindow).Take(_historyWindow).ToList();

                // Extrai as features dessa janela
                var inputs = ExtractFeatures(windowHistory);

                // Cria o vetor de 60 saídas (0..1). 1 se esse número saiu no sorteio i, senão 0
                var outputs = new double[NUMBER_RANGE];
                Array.Fill(outputs, 0.0);

                var draw = history[i]; // o sorteio "atual"
                outputs[draw.V1 - 1] = 1;
                outputs[draw.V2 - 1] = 1;
                outputs[draw.V3 - 1] = 1;
                outputs[draw.V4 - 1] = 1;
                outputs[draw.V5 - 1] = 1;
                outputs[draw.V6 - 1] = 1;

                inputList.Add(inputs);
                outputList.Add(outputs);
            }

            return new BasicMLDataSet(inputList.ToArray(), outputList.ToArray());
        }

        /// <summary>
        /// Converte as 60 probabilidades em 6 números finais.
        /// Agora não há fallback aleatório: apenas escolhemos 
        /// os 6 índices de maior probabilidade.
        /// </summary>
        private List<int> ConvertPredictionToNumbers(double[] prediction)
        {
            // Cria uma lista com (numero, prob)
            var probList = new List<(int num, double prob)>();
            for (int i = 0; i < NUMBER_RANGE; i++)
            {
                probList.Add((i + 1, prediction[i])); 
            }

            // Ordena do maior para o menor
            var ordered = probList.OrderByDescending(x => x.prob).ToList();

            // Pega os 6 melhores
            var top6 = ordered.Take(6).Select(x => x.num).OrderBy(x => x).ToList();

            return top6;
        }
    }
}