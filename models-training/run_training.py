#!/usr/bin/env python3
"""
Script principal para executar treinos dos modelos de sentiment analysis
"""

import sys
import argparse
from train_baseline import train_baseline_model
from train_production import train_production_model

def main():
    parser = argparse.ArgumentParser(description='Treinar modelos de sentiment analysis')
    parser.add_argument('--model', choices=['baseline', 'production', 'both'],
                       default='baseline', help='Modelo para treinar')
    parser.add_argument('--data', default='datasets/Reviews.csv',
                       help='Caminho para o arquivo de dados')

    args = parser.parse_args()

    print(f"🚀 Iniciando treino do(s) modelo(s): {args.model}")
    print(f"📁 Dados: {args.data}")
    print("-" * 50)

    if args.model == 'baseline' or args.model == 'both':
        print("\n🔹 Treinando modelo BASELINE (DNN Pool)...")
        try:
            model, history, threshold = train_baseline_model(args.data)
            print("✅ Treino do modelo baseline concluído!")
            print(f"📊 Threshold ótimo: {threshold:.3f}")
        except Exception as e:
            print(f"❌ Erro no treino do baseline: {e}")
            if args.model == 'baseline':
                return 1

    if args.model == 'production' or args.model == 'both':
        print("\n🔹 Treinando modelo PRODUÇÃO (BiLSTM)...")
        try:
            model, history, threshold = train_production_model(args.data)
            print("✅ Treino do modelo de produção concluído!")
            print(f"📊 Threshold ótimo: {threshold:.3f}")
        except Exception as e:
            print(f"❌ Erro no treino de produção: {e}")
            if args.model == 'production':
                return 1

    print("\n🎉 Treino(s) finalizado(s) com sucesso!")
    print("📈 Confira os resultados no MLflow: https://mlflow-server-273169854208.us-central1.run.app")
    return 0

if __name__ == "__main__":
    sys.exit(main())