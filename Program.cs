using System;
using System.Collections.Generic;
using System.IO;
using Megasena;
using Megasena.Enhanced;

namespace LottoPredictor
{
    class Program
    {
        private const string DATASET_FILE = "MegaSenaDataSet_old.txt";
        private const int OPTIMAL_DEEPNESS = 50;

        static void Main(string[] args)
        {
            Console.WriteLine("\nQual modo de predição deseja usar?\n");
            Console.WriteLine("=======================\n");
            Console.WriteLine("1 - Modo Original\n");
            Console.WriteLine("2 - Modo Aprimorado (Recomendado)\n");
            var predictionMode = Console.ReadLine();
            var useEnhanced = predictionMode == "2";

            Console.WriteLine("\nQuantos jogos deseja gerar?\n");
            var qtdJogos = Convert.ToInt32(Console.ReadLine());

            MegasenaExecutor.Run(qtdJogos, DATASET_FILE, OPTIMAL_DEEPNESS, useEnhanced);
        }   
    }
}