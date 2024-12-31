using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Megasena.Enhanced;
using Newtonsoft.Json;

namespace Megasena 
{
    public static class MegasenaPredictor
    {
        public static bool CreateDatabase(string fileDB, out MegasenaListResult dbl)
        {
            dbl = new MegasenaListResult();

            using (var reader = File.OpenText(fileDB))
            {
                var line = string.Empty;
                while ((line = reader.ReadLine()) != null)
                {
                    var values = line.Split(' ')[2].Split(',');
                    var res = new MegasenaResult(
                    int.Parse(values[0]),
                    int.Parse(values[1]),
                    int.Parse(values[2]),
                    int.Parse(values[3]),
                    int.Parse(values[4]),
                    int.Parse(values[5])
                    );
                    dbl.Add(res);
                }
            }

            dbl.Reverse();

            return true;
        }

        public static string TrainModel(MegasenaListResult dbl, int deepness, bool useEnhanced = true)
        {
            if (useEnhanced)
            {
                try
                {
                    var enhancedPredictor = new EnhancedPredictor();
                    var predictedNumbers = enhancedPredictor.PredictNextNumbers(dbl.ToList());
                    return string.Join(",", predictedNumbers);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Erro no preditor aprimorado: {ex.Message}");
                    Console.WriteLine($"\n\n {JsonConvert.SerializeObject(ex, Formatting.Indented)} \n\n");
                    Console.WriteLine("Utilizando preditor original como fallback...");
                    return TrainModelOriginal(dbl, deepness);
                }
            }
            else
            {
                return TrainModelOriginal(dbl, deepness);
            }
        }

        public static String TrainModelOriginal(MegasenaListResult dbl, int deepness)
        {
            var deep = deepness;
            var network = new BasicNetwork();

            network.AddLayer(new BasicLayer(null, true, 6 * deep));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5 * 6 * deep));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5 * 6 * deep));
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 6));

            network.Structure.FinalizeStructure();

            var learningInput = new double[deep][];

            for (int i = 0; i < deep; ++i)
            {
                learningInput[i] = new double[deep * 6];

                for (int j = 0, k = 0; j < deep; ++j)
                {
                    var idx = 2 * deep - i - j;
                    var data = dbl[idx];
                    learningInput[i][k++] = (double)data.V1;
                    learningInput[i][k++] = (double)data.V2;
                    learningInput[i][k++] = (double)data.V3;
                    learningInput[i][k++] = (double)data.V4;
                    learningInput[i][k++] = (double)data.V5;
                    learningInput[i][k++] = (double)data.V6;
                }
            }

            var learningOutput = new double[deep][];

            for (int i = 0; i < deep; ++i)
            {
                var idx = deep - 1 - i;
                var data = dbl[idx];

                learningOutput[i] = new double[6]
                {
                            (double)data.V1,
                            (double)data.V2,
                            (double)data.V3,
                            (double)data.V4,
                            (double)data.V5,
                            (double)data.V6
                };
            }

            var trainingSet = new BasicMLDataSet(learningInput, learningOutput);
            var train = new ResilientPropagation(network, trainingSet);
            train.NumThreads = Environment.ProcessorCount;

            START:
            network.Reset();

            RETRY:
            var step = 0;
            
            do
            {
                train.Iteration();
                Console.WriteLine("Train Error: {0}", train.Error);
                ++step;
            }
            while (train.Error > 0.001 && step < 20);

            var passedCount = 0;

            for (var i = 0; i < deep; ++i)
            {
                var should = new MegasenaResult(learningOutput[i]);
                var inputn = new BasicMLData(6 * deep);

                Array.Copy(learningInput[i], inputn.Data, inputn.Data.Length);

                var comput = new MegasenaResult(((BasicMLData)network.Compute(inputn)).Data);
                var passed = should.ToString() == comput.ToString();

                if (passed)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    ++passedCount;
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                }

                Console.WriteLine("{0} {1} {2} {3}",
                should.ToString().PadLeft(17, ' '),
                passed ? "==" : "!=",
                comput.ToString().PadRight(17, ' '),
                passed ? "PASS" : "FAIL");
                Console.ResetColor();
            }

            var input = new BasicMLData(6 * deep);

            for (int i = 0, k = 0; i < deep; ++i)
            {
                var idx = deep - 1 - i;
                var data = dbl[idx];
                input.Data[k++] = (double)data.V1;
                input.Data[k++] = (double)data.V2;
                input.Data[k++] = (double)data.V3;
                input.Data[k++] = (double)data.V4;
                input.Data[k++] = (double)data.V5;
                input.Data[k++] = (double)data.V6;
            }

            var perfect = dbl[0];
            var predict = new MegasenaResult(((BasicMLData)network.Compute(input)).Data);

            Console.ForegroundColor = ConsoleColor.Yellow;
            //Console.WriteLine("Predict: {0}", predict);
            Console.ResetColor();

            if (predict.IsOut())
                goto START;
            if ((double)passedCount < (deep * (double)9 / (double)10) ||
                !predict.IsValid())
                goto RETRY;

            //Console.WriteLine("Press any key for close...");
            //Console.ReadKey(true);
            
            return ReturnOrderedPredictResult(predict.ToString());
        }
        
        public static String ReturnOrderedPredictResult(String predict)
        {
            var numbers = predict.Split(',').ToList();
            numbers.Sort();
            string result = string.Empty;
            foreach(var n in numbers)
            {
                result += n + ",";
            }
            return result.Remove(result.LastIndexOf(','));
        }
    }
        
}
