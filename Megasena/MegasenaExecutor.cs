using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Megasena.Enhanced;

namespace Megasena
{
    public static class MegasenaExecutor
    {
        public static void Run(int qtdJogos, string datasetFile, int deepness, bool useEnhanced = true)
        {
            var fileDB = Path.GetTempFileName();
            var trainningDataset = Path.Combine(Environment.CurrentDirectory, "Megasena/" + datasetFile);
            List<String> predictResults = new List<string>();

            try
            {               
                using (FileStream fs = File.OpenWrite(fileDB))
                {
                    File.OpenRead(trainningDataset).CopyTo(fs);
                }

                MegasenaListResult dbl = null;

                if (MegasenaPredictor.CreateDatabase(fileDB, out dbl))
                {
                    Console.WriteLine("\nIniciando processo de predição...");
                    Console.WriteLine($"Modo: {(useEnhanced ? "Aprimorado" : "Original")}");
                    Console.WriteLine($"Profundidade: {deepness}");
                    Console.WriteLine($"Quantidade de jogos: {qtdJogos}\n");

                    for (int i = 0; i < qtdJogos; i++)
                    {
                        Console.WriteLine($"Processando jogo {i + 1} de {qtdJogos}...");
                        predictResults.Add(MegasenaPredictor.TrainModel(dbl, deepness, useEnhanced));
                    }

                    Console.WriteLine("\n\n================================================================");
                    Console.Write("          Algoritmo Preditivo de Números da MegaSena\n                Made with");
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Write(" S2 ");
                    Console.ResetColor();
                    Console.Write("by Rodrigo Abib\n");
                    Console.WriteLine("================================================================\n\n");

                    int jogo = 1;

                    foreach (var predict in predictResults)
                    {
                        Console.WriteLine("Jogo " + jogo.ToString().PadLeft(2, '0') + "   ---   " + predict.Replace(",", " - "));
                        jogo++;
                    }

                    if (useEnhanced)
                    {
                        Console.WriteLine("\nObservações do modo aprimorado:");
                        Console.WriteLine("- Os números foram gerados utilizando técnicas avançadas de machine learning");
                        Console.WriteLine("- O modelo considera padrões históricos e distribuição estatística");
                        Console.WriteLine("- A precisão pode variar dependendo da qualidade do dataset\n");
                    }

                    Console.WriteLine("\nPressione qualquer tecla para continuar...");
                    Console.ReadLine();
                }
            }
            catch (Exception exception)
            {
                Console.WriteLine($"\nErro durante a execução: {exception.Message}");
                Console.WriteLine("Stack Trace:");
                Console.WriteLine(exception.StackTrace);
            }
            finally
            {
                if (File.Exists(fileDB))
                {
                    File.Delete(fileDB);
                }
            }
        }
    }
}